"""AI Methods for Science (77222), week 2 lab: the same MLP on CIFAR-10, with the week-2 knobs.

Same interface as session 2's mlp_lab.py, same defaults (plain SGD, lr 0.05): every E1-E11 number
carries over unchanged. New hyperparameters: optimizer, momentum, nesterov, schedule, warmup_epochs,
init, batchnorm, augment.

    from mlp_lab2 import train
    result = train({"optimizer": "adam", "lr": 1e-3, "schedule": "cosine", "epochs": 30})
    result["summary"]["val_acc_at_best"]
    result["history"]["val_loss"]          # one entry per epoch

From the shell, a config file in, a result file out (both JSON):

    python mlp_lab2.py --download          # fetch CIFAR-10 (170 MB) into ./data, if you do not have it
    python mlp_lab2.py --show-defaults     # every hyperparameter and its default
    python mlp_lab2.py config.json result.json

Needs Python >= 3.11 with the packages in requirements.txt (torch, torchvision, numpy, pandas,
matplotlib). Install them however you like: pip, conda, uv ("uv run mlp_lab2.py ..." installs them).

Hyperparameters (all optional; missing keys take the defaults below):
    hidden          list of hidden-layer widths, e.g. [512, 256]; [] is a linear classifier
    activation      "relu" | "sigmoid" | "tanh"
    optimizer       "sgd" | "adam" | "adamw"
    lr              (peak) learning rate. Plain SGD wants ~0.05 here; momentum 0.9 ~0.005; Adam ~0.001
    momentum        SGD only: v <- momentum * v + g; theta <- theta - lr * v  (0 = plain SGD)
    nesterov        SGD with momentum only: evaluate the gradient after the momentum step
    weight_decay    L2 penalty (sgd, adam: lambda * theta added to the gradient) or decoupled decay (adamw)
    schedule        "none" | "cosine" (lr -> 0 over the run, per step) | "step" (lr / 10 at 50% and 75%)
    warmup_epochs   linear warmup of the learning rate from 0 over this many epochs (0 = none)
    init            "default" (PyTorch: U(+-1/sqrt(n_in)), Var = 1/(3 n_in)) | "he" (N(0, 2/n_in)) | "xavier" (N(0, 1/n_in))
    batchnorm       BatchNorm1d after every hidden linear layer, before the activation
    dropout         dropout probability after each hidden activation (0 = none); off at evaluation
    augment         random crop (pad 4) + horizontal flip of every training image, fresh every step
    batch_size      minibatch size
    epochs          passes over the training set
    train_fraction  use only this fraction of the 45,000 training images (validation set unchanged)
    normalize       standardize the pixels (true) or feed raw values 0..255 (false)
    seed            initialization, shuffling, dropout and augmentation (the train/validation split is fixed)
    test            also report test accuracy of the best-validation model -- use ONCE, at the end
    device          "cpu" | "cuda" | "mps" | "auto"
    data_dir        where CIFAR-10 is (170 MB, downloaded on the first call)

Metrics returned:
    history  per epoch: train_loss, train_acc, val_loss, val_acc, lr (at the end of the epoch), seconds
    summary  n_params, steps_per_epoch, best_epoch (lowest validation loss), val_loss_at_best,
             val_acc_at_best, max_val_acc, final_train_acc, final_val_acc, final_train_loss,
             final_val_loss, sec_per_epoch, diverged (loss became nan/inf), test_acc (if test)
"""
import copy, json, math, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, Subset, random_split, DataLoader

DEFAULTS = {
    "hidden": [512, 256], "activation": "relu",
    "optimizer": "sgd", "lr": 0.05, "momentum": 0.0, "nesterov": False, "weight_decay": 0.0,
    "schedule": "none", "warmup_epochs": 0,
    "init": "default", "batchnorm": False, "dropout": 0.0, "augment": False,
    "batch_size": 128, "epochs": 20,
    "train_fraction": 1.0, "normalize": True, "seed": 0, "test": False,
    "device": "cpu", "data_dir": "./data",
}
ACTIVATIONS = {"relu": F.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}

# ---- data -------------------------------------------------------------------------
MEAN = torch.tensor([0.4914, 0.4822, 0.4465])        # per-channel mean of the training set
STD  = torch.tensor([0.2470, 0.2435, 0.2616])        # per-channel standard deviation
_cache = {}

def load_cifar10(data_dir="./data", normalize=True):
    """Train / validation / test sets as in-RAM tensors (N, 3, 32, 32). Split 45,000 / 5,000 / 10,000."""
    key = (data_dir, normalize)
    if key not in _cache:
        import torchvision
        def to_tensors(ds):
            x = torch.tensor(ds.data).float()         # (N, 32, 32, 3), values 0..255
            if normalize:
                x = (x / 255 - MEAN) / STD            # [0, 1], then standardize each channel
            return x.permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets)
        x, y = to_tensors(torchvision.datasets.CIFAR10(data_dir, train=True, download=True))
        x_test, y_test = to_tensors(torchvision.datasets.CIFAR10(data_dir, train=False, download=True))
        train, val = random_split(TensorDataset(x, y), [45000, 5000],
                                  generator=torch.Generator().manual_seed(0))
        _cache[key] = (train, val, TensorDataset(x_test, y_test))
    return _cache[key]

def augment_batch(x, pad=4):
    """Random crop after zero-padding by `pad`, then a horizontal flip with probability 1/2, per image."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))                               # (B, C, H+2p, W+2p)
    dy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    dx = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (torch.arange(H, device=x.device)[None, :, None] + dy[:, None, None])   # (B, H, 1)
    cols = (torch.arange(W, device=x.device)[None, None, :] + dx[:, None, None])   # (B, 1, W)
    b = torch.arange(B, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[b, rows, cols].permute(0, 3, 1, 2)   # (B, C, H, W)
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out.contiguous()

# ---- model ------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=3 * 32 * 32, hidden=(512, 256), n_classes=10, activation="relu",
                 dropout=0.0, batchnorm=False, init="default"):
        super().__init__()
        self.drop = nn.Dropout(dropout)               # identity when dropout == 0; off in eval mode
        dims = (in_dim, *hidden)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(d) if batchnorm else nn.Identity() for d in hidden])
        self.out = nn.Linear(dims[-1], n_classes)
        self.phi = ACTIVATIONS[activation]
        if init != "default":                         # PyTorch's default is U(-1/sqrt(n_in), 1/sqrt(n_in))
            for layer in (*self.layers, self.out):
                fan_in = layer.weight.shape[1]
                std = {"he": math.sqrt(2 / fan_in), "xavier": math.sqrt(1 / fan_in)}[init]
                nn.init.normal_(layer.weight, 0.0, std)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.flatten(1)                              # (B, 3, 32, 32) -> (B, 3072)
        for layer, norm in zip(self.layers, self.norms):
            x = self.drop(self.phi(norm(layer(x))))   # linear -> (batchnorm) -> activation -> (dropout)
        return self.out(x)                            # logits; the softmax lives inside the loss

def evaluate(model, loader, device):
    """Mean cross-entropy and accuracy of the model on a data set."""
    model.eval()
    loss = correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return loss / n, correct / n

# ---- optimizer and schedule ------------------------------------------------------
def make_optimizer(model, cfg):
    params, lr, wd = model.parameters(), cfg["lr"], cfg["weight_decay"]
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg["momentum"], nesterov=cfg["nesterov"], weight_decay=wd)
    if cfg["optimizer"] == "adam":                    # weight_decay here is L2 in the gradient (coupled)
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if cfg["optimizer"] == "adamw":                   # decoupled: theta <- theta - lr * wd * theta, then the Adam step
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer {cfg['optimizer']!r}")

def lr_factor(step, total_steps, warmup_steps, schedule):
    """Multiplier of the peak learning rate at gradient step `step` (0-based)."""
    f = min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0
    if schedule == "cosine":
        f *= 0.5 * (1 + math.cos(math.pi * step / total_steps))
    elif schedule == "step":
        f *= 0.1 ** sum(step >= m * total_steps for m in (0.5, 0.75))
    elif schedule != "none":
        raise ValueError(f"unknown schedule {schedule!r}")
    return f

# ---- training ---------------------------------------------------------------------
def train(config=None, verbose=True):
    """Train an MLP with the given hyperparameters (dict; see the module docstring). Returns
    {"config": ..., "history": ..., "summary": ...}, all plain Python types (JSON-serializable)."""
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

    model = MLP(hidden=tuple(cfg["hidden"]), activation=cfg["activation"], dropout=cfg["dropout"],
                batchnorm=cfg["batchnorm"], init=cfg["init"]).to(device)
    optimizer = make_optimizer(model, cfg)
    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = cfg["warmup_epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup_steps, cfg["schedule"]))
    loss_fn = nn.CrossEntropyLoss()                   # log-softmax + negative log-likelihood
    n_params = sum(q.numel() for q in model.parameters())
    if verbose:
        print(f"MLP {'-'.join(map(str, [3072, *cfg['hidden'], 10]))} ({cfg['activation']}"
              f"{', batchnorm' if cfg['batchnorm'] else ''}): {n_params:,} parameters, "
              f"{len(train_loader)} steps per epoch; {cfg['optimizer']} lr {cfg['lr']:g}"
              f"{', momentum ' + str(cfg['momentum']) if cfg['momentum'] else ''}"
              f"{', ' + cfg['schedule'] if cfg['schedule'] != 'none' else ''}", flush=True)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds")}
    best = (math.inf, None, 0)                        # (validation loss, parameters, epoch)
    diverged = False
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time()
        run_loss = run_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if cfg["augment"]:
                x = augment_batch(x)                  # a fresh random crop and flip of every image
            optimizer.zero_grad()                     # .grad accumulates; clear it
            logits = model(x)                         # forward pass on a minibatch
            loss = loss_fn(logits, y)                 # a scalar
            loss.backward()                           # backprop: fills every parameter's .grad
            optimizer.step()                          # the optimizer's update rule
            scheduler.step()                          # the learning rate for the next step
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
        val_loss, val_acc = evaluate(model, val_loader, device)
        row = (run_loss / len(train_set), run_correct / len(train_set), val_loss, val_acc,
               optimizer.param_groups[0]["lr"], time.time() - t0)
        for k, v in zip(history, row):
            history[k].append(v)
        if verbose:
            print(f"epoch {epoch:3d}   train loss {row[0]:.3f} acc {row[1]:.3f}   "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f}   lr {row[4]:.2e}   {row[5]:.1f}s", flush=True)
        if not math.isfinite(row[0]):
            diverged = True
            if verbose:
                print("loss is not finite: stopping", flush=True)
            break
        if val_loss < best[0]:                        # early stopping: remember the best epoch
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
    if cfg["test"] and best[2]:                       # the test set: touched once, at the end
        model.load_state_dict(best[1])
        summary["test_acc"] = evaluate(model, DataLoader(test_set, batch_size=1000), device)[1]
        if verbose:
            print(f"test accuracy (epoch-{best[2]} model): {summary['test_acc']:.3f}", flush=True)
    # nan/inf are not valid JSON: store them as None
    clean = lambda v: v if not isinstance(v, float) or math.isfinite(v) else None
    history = {k: [clean(v) for v in vs] for k, vs in history.items()}
    summary = {k: clean(v) for k, v in summary.items()}
    return {"config": cfg, "history": history, "summary": summary}

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--download":
        load_cifar10(); print("CIFAR-10 is in ./data")
    elif len(sys.argv) == 2 and sys.argv[1] == "--show-defaults":
        print(json.dumps(DEFAULTS, indent=1))
    elif len(sys.argv) == 3:
        result = train(json.load(open(sys.argv[1])))
        json.dump(result, open(sys.argv[2], "w"), indent=1)
        print(f"result written to {sys.argv[2]}")
    else:
        print(__doc__)
