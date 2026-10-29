"""AI Methods for Science (77222), session 5 lab: a convolutional network on CIFAR-10, same loop as the MLP.

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
    skip            add a skip connection around every conv (identity, or a 1x1 projection where the channel count
                    changes): the residual blocks of ResNet. For E22.
    batchnorm       BatchNorm after every conv (true) or not
    hidden          the MLP's hidden widths, used only when channels == []
    shuffle_pixels  apply one fixed random permutation to the 3072 pixels of every image (train and val):
                    destroys the spatial structure, keeps every pixel value. For E16.
    classes         which CIFAR-10 classes to use, e.g. [0, 1, 2, 3, 4]; [] = all ten. Train, validation and
                    test sets are restricted to them and the labels renumbered 0..K-1. For E20.
    n_per_class     keep only this many training images per class (0 = all). For E20.
    save_weights    path; after training, save the weights of the best epoch there (torch.save). For E20.
    init_from       path to weights saved by save_weights: the conv blocks start from them, the linear head is
                    new (sized to the current classes). The architecture keys must match. For E20.
    freeze          with init_from: train only the head, the conv blocks stay as loaded. For E20.
    nearest_mean(weights, classes, n_per_class): no training at all, nearest class mean on the saved net's features. E20.
    optimizer, lr, momentum, weight_decay, schedule, warmup_epochs, dropout, augment, batch_size, epochs,
    train_fraction, normalize, seed, test, device, data_dir   as in mlp_lab2.py

Defaults: channels [32, 64, 128], one conv per block, BatchNorm on, AdamW lr 1e-3, weight decay 0.05,
cosine schedule, augmentation on, 15 epochs: the "recipe" of session 3 applied to a small CNN.
Metrics returned: as in mlp_lab2.py, plus result["model"]: the trained network (best epoch, on the CPU).

plot_first_layer(result, path=...) draws the first-layer stencils (at initialization and after training) and their
feature maps on one validation image. Not part of the lab: with this recipe the first layer stays close to its
random initialization (cosine similarity 0.85-0.99 per stencil after 10 epochs), so there is little to see at 3x3.
"""
import copy, json, math, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from mlp_lab2 import MLP, augment_batch, evaluate, load_cifar10, lr_factor, make_optimizer

DEFAULTS = {
    "channels": [32, 64, 128], "convs_per_block": 1, "kernel": 3, "batchnorm": True,
    "hidden": [512, 256], "shuffle_pixels": False, "skip": False,
    "classes": [], "n_per_class": 0, "save_weights": "", "init_from": "", "freeze": False,
    "optimizer": "adamw", "lr": 1e-3, "momentum": 0.0, "nesterov": False, "weight_decay": 0.05,
    "schedule": "cosine", "warmup_epochs": 0, "dropout": 0.0, "augment": True,
    "batch_size": 128, "epochs": 15,
    "train_fraction": 1.0, "normalize": True, "seed": 0, "test": False,
    "device": "auto", "data_dir": "./data",
}

class ConvUnit(nn.Module):
    """conv -> (BatchNorm) -> (+ skip) -> ReLU. The skip adds the unit's input to its output (a 1x1 projection when
    the number of channels changes): the unit then learns a residual, F(x) = output - x. He et al. 2016."""
    def __init__(self, c_in, c_out, kernel=3, batchnorm=True, skip=False):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2)              # same spatial size
        self.bn = nn.BatchNorm2d(c_out) if batchnorm else nn.Identity()
        self.skip = skip
        self.proj = (nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)) if skip else None

    def forward(self, x):
        y = self.bn(self.conv(x))
        if self.skip:
            y = y + self.proj(x)
        return F.relu(y)

class CNN(nn.Module):
    def __init__(self, channels=(32, 64, 128), convs_per_block=1, kernel=3, batchnorm=True, n_classes=10, dropout=0.0, skip=False):
        super().__init__()
        layers, c_in = [], 3
        for c_out in channels:
            for _ in range(convs_per_block):
                layers.append(ConvUnit(c_in, c_out, kernel, batchnorm, skip))
                c_in = c_out
            layers.append(nn.MaxPool2d(2))                                            # 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(c_in, n_classes)

    def forward(self, x):
        x = self.features(x)                          # (B, C, H, W)
        x = x.mean(dim=(2, 3))                        # global average pooling -> (B, C)
        return self.out(self.drop(x))

ARCH_KEYS = ("channels", "convs_per_block", "kernel", "batchnorm", "skip")

def build_cnn(cfg, n_classes=10):
    return CNN(tuple(cfg["channels"]), cfg["convs_per_block"], cfg["kernel"], cfg["batchnorm"],
               n_classes=n_classes, dropout=cfg.get("dropout", 0.0), skip=cfg["skip"])

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
    n_classes = 10
    if cfg["classes"]:
        train_set, val_set, test_set = (restrict(ds, cfg["classes"]) for ds in (train_set, val_set, test_set))
        n_classes = len(cfg["classes"])
    if cfg["n_per_class"]:
        train_set = per_class_subset(train_set, cfg["n_per_class"], n_classes, cfg["seed"])
    if cfg["train_fraction"] < 1:
        train_set = Subset(train_set, range(int(cfg["train_fraction"] * len(train_set))))
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000)
    perm = torch.randperm(3 * 32 * 32, generator=torch.Generator().manual_seed(1)) if cfg["shuffle_pixels"] else None
    def shuffle(x):                                   # the same permutation for every image, every time
        return x.flatten(1)[:, perm].view_as(x) if perm is not None else x

    if cfg["channels"]:
        model = build_cnn(cfg, n_classes)
    else:
        model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"])
        if n_classes != 10:
            model.out = nn.Linear(model.out.in_features, n_classes)
    if cfg["init_from"]:                              # transfer: the conv blocks from a saved net, a new head
        saved = torch.load(cfg["init_from"], map_location="cpu")
        features = {k[len("features."):]: v for k, v in saved["state_dict"].items() if k.startswith("features.")}
        model.features.load_state_dict(features)
        if cfg["freeze"]:
            model.features.requires_grad_(False)
    model = model.to(device)
    optimizer = make_optimizer(model, cfg)
    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = cfg["warmup_epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup_steps, cfg["schedule"]))
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)   # trainable parameters
    if verbose:
        arch = f"CNN {cfg['channels']}x{cfg['convs_per_block']}{' +skip' if cfg['skip'] else ''}" if cfg["channels"] else f"MLP {cfg['hidden']}"
        print(f"{arch}: {n_params:,} parameters, {len(train_loader)} steps per epoch; {cfg['optimizer']} lr {cfg['lr']:g}"
              f"{', ' + cfg['schedule'] if cfg['schedule'] != 'none' else ''}{', augment' if cfg['augment'] else ''}"
              f"{', shuffled pixels' if perm is not None else ''}"
              f"{', classes ' + str(cfg['classes']) if cfg['classes'] else ''}"
              f"{', ' + str(len(train_set)) + ' training images' if cfg['n_per_class'] else ''}"
              f"{', from ' + cfg['init_from'] + (' (frozen)' if cfg['freeze'] else ' (fine-tuned)') if cfg['init_from'] else ''}"
              f"; device {device}", flush=True)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds")}
    best = (math.inf, None, 0); diverged = False
    val_eval = [(shuffle(x), y) for x, y in val_loader]          # shuffled once if asked, else as is
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time()
        if cfg["init_from"] and cfg["freeze"]:
            model.features.eval()                     # frozen features: BatchNorm uses its stored statistics
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
    if best[2]:
        model.load_state_dict(best[1])                # the model of the best epoch, from here on
    if cfg["test"] and best[2]:
        test_eval = [(shuffle(x), y) for x, y in DataLoader(test_set, batch_size=1000)]
        summary["test_acc"] = evaluate(model, test_eval, device)[1]
        if verbose:
            print(f"test accuracy (epoch-{best[2]} model): {summary['test_acc']:.3f}", flush=True)
    model = model.cpu().eval()
    if cfg["save_weights"]:
        torch.save({"state_dict": model.state_dict(), "arch": {k: cfg[k] for k in ARCH_KEYS}}, cfg["save_weights"])
        if verbose:
            print(f"weights saved to {cfg['save_weights']}", flush=True)
    clean = lambda v: v if not isinstance(v, float) or math.isfinite(v) else None
    history = {k: [clean(v) for v in vs] for k, vs in history.items()}
    summary = {k: clean(v) for k, v in summary.items()}
    return {"config": cfg, "history": history, "summary": summary, "model": model}

def restrict(dataset, classes):
    """The subset of a (Subset of a) TensorDataset with labels in `classes`, labels renumbered 0..K-1."""
    xs, ys = zip(*[dataset[i] for i in range(len(dataset))])
    x, y = torch.stack(xs), torch.stack(ys)
    lookup = torch.full((10,), -1, dtype=torch.long)
    lookup[torch.tensor(classes)] = torch.arange(len(classes))
    keep = lookup[y] >= 0
    return TensorDataset(x[keep], lookup[y[keep]])

def per_class_subset(dataset, n, n_classes, seed):
    """`n` training images of each class, chosen at random (fixed by `seed`)."""
    y = torch.stack([dataset[i][1] for i in range(len(dataset))])
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in range(n_classes):
        where = (y == c).nonzero().flatten()
        idx.append(where[torch.randperm(len(where), generator=g)[:n]])
    return Subset(dataset, torch.cat(idx).tolist())

@torch.no_grad()
def nearest_mean(weights, classes, n_per_class, data_dir="./data", seed=0, device="auto"):
    """No training at all: the conv blocks of a saved net (save_weights) turn every image into 128 numbers; each
    class of `classes` gets the mean of its n_per_class labelled training images; a validation image is assigned
    to the nearest mean. Returns {"val_acc": ..., "n_train": ...}. For E20 (b)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    saved = torch.load(weights, map_location="cpu")
    model = build_cnn(saved["arch"])
    model.features.load_state_dict({k[len("features."):]: v for k, v in saved["state_dict"].items() if k.startswith("features.")})
    model = model.to(device).eval()
    train_set, val_set, _ = load_cifar10(data_dir, True)
    train_set, val_set = restrict(train_set, classes), restrict(val_set, classes)
    train_set = per_class_subset(train_set, n_per_class, len(classes), seed)
    def feats(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))]); ys = torch.stack([ds[i][1] for i in range(len(ds))])
        f = torch.cat([model.features(xs[i:i + 500].to(device)).mean(dim=(2, 3)).cpu() for i in range(0, len(xs), 500)])
        return f, ys
    ft, yt = feats(train_set); fv, yv = feats(val_set)
    means = torch.stack([ft[yt == c].mean(0) for c in range(len(classes))])          # one prototype per class
    pred = torch.cdist(fv, means).argmin(1)
    return {"val_acc": (pred == yv).float().mean().item(), "n_train": len(train_set)}

@torch.no_grad()
def plot_first_layer(result, image_index=0, path=None):
    """Three rows: the stencils of the first conv layer at initialization (same seed, before any training), the same
    stencils after training, and the feature maps the trained ones produce on one validation image. One tile per
    output channel; the 3 input channels of a stencil are shown as RGB, each tile scaled to its own range; feature
    maps in gray, white = large (after the conv, before BatchNorm and ReLU). Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    cfg, model = result["config"], result["model"]
    conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), None)
    if conv is None:
        raise ValueError("no conv layer: this is the MLP")
    torch.manual_seed(cfg["seed"])                                        # the initialization train() started from
    init = build_cnn(cfg)
    w0 = next(m for m in init.modules() if isinstance(m, nn.Conv2d)).weight.detach()
    w = conv.weight.detach().cpu()                                        # (C_out, 3, k, k)
    _, val_set, _ = load_cifar10(cfg["data_dir"], cfg["normalize"])
    x, y = val_set[image_index]
    if cfg["shuffle_pixels"]:
        perm = torch.randperm(3 * 32 * 32, generator=torch.Generator().manual_seed(1))
        x = x.flatten()[perm].view_as(x)
    maps = F.conv2d(x[None], w, conv.bias, padding=conv.padding)[0]      # (C_out, 32, 32)
    C = w.shape[0]; cols = min(C, 16); rows = math.ceil(C / cols)
    fig, axs = plt.subplots(3 * rows + 1, cols, figsize=(cols * 0.8, (3 * rows + 1) * 0.9))
    axs = axs.reshape(3 * rows + 1, cols)
    for ax in axs.flat:
        ax.axis("off")
    img = x * torch.tensor([0.2470, 0.2435, 0.2616])[:, None, None] + torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None]
    axs[0, 0].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest"); axs[0, 0].set_title("input", fontsize=7)
    fig.suptitle(f"{C} first-layer stencils {tuple(w.shape[1:])}: at initialization (rows 1), after training (rows 2), "
                 "each tile scaled to its own range; their feature maps on the input (rows 3, white = large)", fontsize=7)
    tile = lambda t: ((t - t.min()) / (t.max() - t.min() + 1e-8)).permute(1, 2, 0).numpy()
    for c in range(C):
        r, col = divmod(c, cols)
        axs[1 + r, col].imshow(tile(w0[c]), interpolation="nearest")
        axs[1 + rows + r, col].imshow(tile(w[c]), interpolation="nearest")
        axs[1 + 2 * rows + r, col].imshow(maps[c].numpy(), cmap="gray", interpolation="nearest")
    fig.tight_layout(pad=0.2)
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--show-defaults":
        print(json.dumps(DEFAULTS, indent=1))
    elif len(sys.argv) == 3:
        result = train(json.load(open(sys.argv[1])))
        result.pop("model")                           # not JSON; use save_weights to keep the weights
        json.dump(result, open(sys.argv[2], "w"), indent=1)
        print(f"result written to {sys.argv[2]}")
    else:
        print(__doc__)
