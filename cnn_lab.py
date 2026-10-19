"""AI Methods for Science (77222), session 4 lab: a convolutional network on CIFAR-10, same loop as the MLP.

Same interface as mlp_lab2.py: train(config) -> {"config", "history", "summary"}. The ansatz is the only
thing that changed: a stack of conv blocks (conv 3x3 -> BatchNorm -> ReLU, optionally twice) with 2x2
max-pooling between them, global average pooling, and one linear layer. "channels": [] gives the MLP of
session 2 (hidden [512, 256]) for a like-for-like comparison inside the same script.

    from cnn_lab import train
    result = train({"channels": [32, 64, 128], "epochs": 15})
    result["summary"]["val_acc_at_best"], result["summary"]["n_params"]

    python cnn_lab.py --show-defaults
    python cnn_lab.py config.json result.json

Hyperparameters (all optional):
    channels        output channels of each conv block, e.g. [32, 64, 128]; [] = the session-2 MLP
    convs_per_block 1 or 2 conv layers per block (VGG-style)
    kernel          kernel size of every conv (3)
    batchnorm       BatchNorm after every conv (true) or not
    hidden          the MLP's hidden widths, used only when channels == []
    shuffle_pixels  apply one fixed random permutation to the 3072 pixels of every image (train and val):
                    destroys the spatial structure, keeps every pixel value. For E16.
    optimizer, lr, momentum, weight_decay, schedule, warmup_epochs, dropout, augment, batch_size, epochs,
    train_fraction, normalize, seed, test, device, data_dir   as in mlp_lab2.py

Defaults: channels [32, 64, 128], one conv per block, BatchNorm on, AdamW lr 1e-3, weight decay 0.05,
cosine schedule, augmentation on, 15 epochs: the "recipe" of session 3 applied to a small CNN.
Metrics returned: as in mlp_lab2.py.
"""
import copy, json, math, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from mlp_lab2 import MLP, augment_batch, evaluate, load_cifar10, lr_factor, make_optimizer

DEFAULTS = {
    "channels": [32, 64, 128], "convs_per_block": 1, "kernel": 3, "batchnorm": True,
    "hidden": [512, 256], "shuffle_pixels": False,
    "optimizer": "adamw", "lr": 1e-3, "momentum": 0.0, "nesterov": False, "weight_decay": 0.05,
    "schedule": "cosine", "warmup_epochs": 0, "dropout": 0.0, "augment": True,
    "batch_size": 128, "epochs": 15,
    "train_fraction": 1.0, "normalize": True, "seed": 0, "test": False,
    "device": "auto", "data_dir": "./data",
}

class CNN(nn.Module):
    def __init__(self, channels=(32, 64, 128), convs_per_block=1, kernel=3, batchnorm=True, n_classes=10, dropout=0.0):
        super().__init__()
        layers, c_in = [], 3
        for c_out in channels:
            for _ in range(convs_per_block):
                layers.append(nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2))   # same spatial size
                if batchnorm:
                    layers.append(nn.BatchNorm2d(c_out))
                layers.append(nn.ReLU())
                c_in = c_out
            layers.append(nn.MaxPool2d(2))                                            # 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(c_in, n_classes)

    def forward(self, x):
        x = self.features(x)                          # (B, C, H, W)
        x = x.mean(dim=(2, 3))                        # global average pooling -> (B, C)
        return self.out(self.drop(x))

def train(config=None, verbose=True):
    cfg = {**DEFAULTS, **(config or {})}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown hyperparameters: {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    train_set, val_set, test_set = load_cifar10(cfg["data_dir"], cfg["normalize"])
    if cfg["train_fraction"] < 1:
        train_set = Subset(train_set, range(int(cfg["train_fraction"] * len(train_set))))
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000)
    perm = torch.randperm(3 * 32 * 32, generator=torch.Generator().manual_seed(1)) if cfg["shuffle_pixels"] else None
    def shuffle(x):                                   # the same permutation for every image, every time
        return x.flatten(1)[:, perm].view_as(x) if perm is not None else x

    if cfg["channels"]:
        model = CNN(tuple(cfg["channels"]), cfg["convs_per_block"], cfg["kernel"], cfg["batchnorm"], dropout=cfg["dropout"])
    else:
        model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"])
    model = model.to(device)
    optimizer = make_optimizer(model, cfg)
    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = cfg["warmup_epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup_steps, cfg["schedule"]))
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(q.numel() for q in model.parameters())
    if verbose:
        arch = f"CNN {cfg['channels']}x{cfg['convs_per_block']}" if cfg["channels"] else f"MLP {cfg['hidden']}"
        print(f"{arch}: {n_params:,} parameters, {len(train_loader)} steps per epoch; {cfg['optimizer']} lr {cfg['lr']:g}"
              f"{', ' + cfg['schedule'] if cfg['schedule'] != 'none' else ''}{', augment' if cfg['augment'] else ''}"
              f"{', shuffled pixels' if perm is not None else ''}; device {device}", flush=True)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds")}
    best = (math.inf, None, 0); diverged = False
    val_eval = [(shuffle(x), y) for x, y in val_loader]          # shuffled once if asked, else as is
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time()
        run_loss = run_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if cfg["augment"]:
                x = augment_batch(x)
            x = shuffle(x)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
        val_loss, val_acc = evaluate(model, val_eval, device)
        row = (run_loss / len(train_set), run_correct / len(train_set), val_loss, val_acc,
               optimizer.param_groups[0]["lr"], time.time() - t0)
        for k, v in zip(history, row):
            history[k].append(v)
        if verbose:
            print(f"epoch {epoch:3d}   train loss {row[0]:.3f} acc {row[1]:.3f}   "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f}   lr {row[4]:.2e}   {row[5]:.1f}s", flush=True)
        if not math.isfinite(row[0]):
            diverged = True; break
        if val_loss < best[0]:
            best = (val_loss, copy.deepcopy(model.state_dict()), epoch)

    secs = sorted(history["seconds"])
    summary = {
        "n_params": n_params, "steps_per_epoch": len(train_loader),
        "best_epoch": best[2],
        "val_loss_at_best": best[0] if best[2] else None,
        "val_acc_at_best": history["val_acc"][best[2] - 1] if best[2] else None,
        "max_val_acc": max(history["val_acc"]),
        "final_train_loss": history["train_loss"][-1], "final_train_acc": history["train_acc"][-1],
        "final_val_loss": history["val_loss"][-1], "final_val_acc": history["val_acc"][-1],
        "sec_per_epoch": secs[len(secs) // 2], "diverged": diverged, "test_acc": None,
    }
    if cfg["test"] and best[2]:
        model.load_state_dict(best[1])
        test_eval = [(shuffle(x), y) for x, y in DataLoader(test_set, batch_size=1000)]
        summary["test_acc"] = evaluate(model, test_eval, device)[1]
        if verbose:
            print(f"test accuracy (epoch-{best[2]} model): {summary['test_acc']:.3f}", flush=True)
    clean = lambda v: v if not isinstance(v, float) or math.isfinite(v) else None
    history = {k: [clean(v) for v in vs] for k, vs in history.items()}
    summary = {k: clean(v) for k, v in summary.items()}
    return {"config": cfg, "history": history, "summary": summary}

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--show-defaults":
        print(json.dumps(DEFAULTS, indent=1))
    elif len(sys.argv) == 3:
        result = train(json.load(open(sys.argv[1])))
        json.dump(result, open(sys.argv[2], "w"), indent=1)
        print(f"result written to {sys.argv[2]}")
    else:
        print(__doc__)
