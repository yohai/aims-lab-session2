"""AI Methods for Science (77222), session 2 lab: an MLP on CIFAR-10.

One function does everything: train(config) -> result.

    from mlp_lab import train
    result = train({"lr": 0.1, "epochs": 10})
    result["summary"]["val_acc_at_best"]
    result["history"]["val_loss"]          # one entry per epoch

From the shell, a config file in, a result file out (both JSON):

    python mlp_lab.py --download           # fetch CIFAR-10 (170 MB) into ./data; do this before class
    python mlp_lab.py --show-defaults      # every hyperparameter and its default
    python mlp_lab.py config.json result.json

Needs Python >= 3.11 with the packages in requirements.txt (torch, torchvision, numpy, pandas,
matplotlib). Install them however you like: pip, conda, uv ("uv run mlp_lab.py ..." installs them).

Hyperparameters (all optional; missing keys take the defaults below):
    hidden          list of hidden-layer widths, e.g. [512, 256]; [] is a linear classifier
    activation      "relu" | "sigmoid" | "tanh"
    lr              learning rate of plain SGD
    weight_decay    L2 penalty: adds (weight_decay/2) * |theta|^2 to the loss (SGD's weight_decay)
    dropout         dropout probability after each hidden activation (0 = none); off at evaluation
    batch_size      minibatch size
    epochs          passes over the training set
    train_fraction  use only this fraction of the 45,000 training images (validation set unchanged)
    normalize       standardize the pixels (true) or feed raw values 0..255 (false)
    seed            initialization and shuffling (the train/validation split is always the same)
    test            also report test accuracy of the best-validation model -- use ONCE, at the end
    device          "cpu" | "cuda" | "mps" | "auto"
    data_dir        where CIFAR-10 is downloaded (170 MB, first call only)

Metrics returned:
    history  per epoch: train_loss, train_acc, val_loss, val_acc, seconds
    summary  n_params, steps_per_epoch, best_epoch (lowest validation loss), val_loss_at_best,
             val_acc_at_best, max_val_acc, final_train_acc, final_val_acc, final_train_loss,
             final_val_loss, sec_per_epoch, diverged (loss became nan/inf), test_acc (if test)
"""
import copy, json, math, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, Subset, random_split, DataLoader

DEFAULTS = {
    "hidden": [512, 256], "activation": "relu", "lr": 0.05, "weight_decay": 0.0, "dropout": 0.0,
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

# ---- model ------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=3 * 32 * 32, hidden=(512, 256), n_classes=10, activation="relu", dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)               # identity when dropout == 0; off in eval mode
        dims = (in_dim, *hidden)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.out = nn.Linear(dims[-1], n_classes)
        self.phi = ACTIVATIONS[activation]

    def forward(self, x):
        x = x.flatten(1)                              # (B, 3, 32, 32) -> (B, 3072)
        for layer in self.layers:
            x = self.drop(self.phi(layer(x)))
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

    model = MLP(hidden=tuple(cfg["hidden"]), activation=cfg["activation"], dropout=cfg["dropout"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()                   # log-softmax + negative log-likelihood
    n_params = sum(q.numel() for q in model.parameters())
    if verbose:
        print(f"MLP {'-'.join(map(str, [3072, *cfg['hidden'], 10]))} ({cfg['activation']}): "
              f"{n_params:,} parameters, {len(train_loader)} steps per epoch", flush=True)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "seconds")}
    best = (math.inf, None, 0)                        # (validation loss, parameters, epoch)
    diverged = False
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time()
        run_loss = run_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()                     # .grad accumulates; clear it
            logits = model(x)                         # forward pass on a minibatch
            loss = loss_fn(logits, y)                 # a scalar
            loss.backward()                           # backprop: fills every parameter's .grad
            optimizer.step()                          # theta <- theta - lr * grad
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
        val_loss, val_acc = evaluate(model, val_loader, device)
        row = (run_loss / len(train_set), run_correct / len(train_set), val_loss, val_acc, time.time() - t0)
        for k, v in zip(history, row):
            history[k].append(v)
        if verbose:
            print(f"epoch {epoch:3d}   train loss {row[0]:.3f} acc {row[1]:.3f}   "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f}   {row[4]:.1f}s", flush=True)
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
