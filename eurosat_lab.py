"""AI Methods for Science (77222), week 3 homework: transfer learning on EuroSAT, same interface as the other labs.

EuroSAT (Helber et al. 2019): 27,000 Sentinel-2 satellite patches, 64x64 RGB, 10 land-use classes. Fixed split:
5,000 test, 2,000 validation, the rest a pool from which `n` labelled training images are drawn (n/10 per class).
Four ways to classify them, chosen by "method":
    mlp        the session-2 MLP (12288-512-256-10) from scratch on the pixels
    scratch    ResNet-18 from random initialization
    frozen     ImageNet-pretrained ResNet-18 as a fixed feature extractor (512 numbers per image) + a linear head
    finetune   ImageNet-pretrained ResNet-18, all weights trained, small learning rate

    from eurosat_lab import train
    result = train({"method": "frozen", "n": 300})
    result["summary"]["val_acc_at_best"], result["summary"]["test_acc"]

The challenge path: build the model yourself and hand it over; "method" is then ignored. Any nn.Module that maps a
(B, 3, size, size) batch to 10 logits:
    from eurosat_lab import train, extract_features, load_eurosat
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Linear(512, 10)
    train({"n": 300, "epochs": 15, "lr": 1e-4}, model=m)                       # fine-tuning, your own build
For a frozen backbone, compute its features once and train your head on them (the backbone is slow on a CPU):
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); backbone.fc = nn.Identity()
    feats = extract_features(backbone, size=128)                               # (27000, 512), all images
    train({"n": 300, "epochs": 100, "lr": 1e-2}, model=nn.Linear(512, 10), feats=feats)

    python eurosat_lab.py --download        # fetch EuroSAT (94 MB) and the ResNet-18 weights (45 MB); do this at home
    python eurosat_lab.py --show-defaults
    python eurosat_lab.py config.json result.json

Hyperparameters (all optional):
    method          "mlp" | "scratch" | "frozen" | "finetune"; ignored when you pass model=
    n               labelled training images, n/10 per class (0 = the whole pool of 20,000)
    epochs          0 = the method's default (mlp 40, scratch 30, frozen 100, finetune 15)
    lr              0 = the method's default (1e-3; finetune 1e-4). AdamW, constant learning rate
    weight_decay    0 = the method's default (0; frozen 1e-2)
    batch_size      0 = the method's default (64; finetune 32)
    size            input resolution for the ResNets: the 64x64 patches are resized to size x size (128 helps the
                    pretrained stencils; 64 is 4x cheaper). The MLP always sees 64x64
    augment         a random horizontal flip per batch (not for frozen features)
    seed, test, device, data_dir   as in the other labs. The split is fixed (seed 0) whatever `seed` is

Metrics returned: as in the other labs ({"config", "history", "summary", "model"}); summary["n_params"] counts the
trainable parameters. Early stopping on the validation loss; the test set once, with "test": true.
Frozen features are computed once per image and cached in data_dir/eurosat_feats_{size}.pt.
"""
import copy, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEFAULTS = {
    "method": "frozen", "n": 300, "epochs": 0, "lr": 0.0, "weight_decay": 0.0, "batch_size": 0, "size": 128,
    "augment": True, "seed": 0, "test": False, "device": "auto", "data_dir": "./data",
}
METHOD_DEFAULTS = {                                   # epochs, lr, weight_decay, batch_size
    "mlp": (40, 1e-3, 0.0, 64), "scratch": (30, 1e-3, 0.0, 64), "frozen": (100, 1e-3, 1e-2, 64), "finetune": (15, 1e-4, 0.0, 32),
    "custom": (20, 1e-3, 0.0, 64),                    # model= given: your own build, your own numbers
}
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # ImageNet statistics: what the pretrained net expects
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_cache = {}

def load_eurosat(data_dir="./data"):
    """All 27,000 images as one uint8 tensor (N, 3, 64, 64), the labels, the class names; and the fixed split."""
    data_dir = Path(data_dir)
    if "data" not in _cache:
        cache = data_dir / "eurosat_uint8.pt"
        if cache.exists():
            X, Y, classes = torch.load(cache)
        else:
            import torchvision
            ds = torchvision.datasets.EuroSAT(str(data_dir), download=True)
            X = torch.stack([torch.from_numpy(np.asarray(img)).permute(2, 0, 1) for img, _ in ds])
            Y = torch.tensor(ds.targets); classes = ds.classes
            torch.save((X, Y, classes), cache)
        perm = torch.from_numpy(np.random.default_rng(0).permutation(len(Y)))
        _cache["data"] = (X, Y, classes, {"test": perm[:5000], "val": perm[5000:7000], "pool": perm[7000:]})
    return _cache["data"]

def prep(xb, size):
    """uint8 (B, 3, 64, 64) -> float, ImageNet-standardized, resized to size x size."""
    x = (xb.float() / 255 - IM_MEAN) / IM_STD
    return x if size == 64 else F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

def pretrained_resnet18(n_classes=10, freeze=False):
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze:
        m.requires_grad_(False)
    m.fc = nn.Linear(512, n_classes)                  # the head: 10 classes instead of ImageNet's 1000
    return m

def frozen_features(idx, size, device, data_dir, verbose):
    """The 512 pretrained features of the images in idx, computed once per image and cached on disk."""
    X = load_eurosat(data_dir)[0]
    path = Path(data_dir) / f"eurosat_feats_{size}.pt"
    feats, done = torch.load(path) if path.exists() else (torch.zeros(len(X), 512), torch.zeros(len(X), dtype=torch.bool))
    todo = idx[~done[idx]]
    if len(todo):
        t0 = time.time()
        backbone = pretrained_resnet18(freeze=True).to(device).eval(); backbone.fc = nn.Identity()
        with torch.no_grad():
            for i in range(0, len(todo), 250):
                b = todo[i:i + 250]
                feats[b] = backbone(prep(X[b], size).to(device)).cpu()
        done[todo] = True
        torch.save((feats, done), path)
        if verbose:
            print(f"frozen features for {len(todo)} new images: {time.time() - t0:.0f} s on {device}", flush=True)
    return feats

@torch.no_grad()
def extract_features(backbone, idx=None, size=128, device="auto", data_dir="./data"):
    """Run `backbone` (a module mapping (B, 3, size, size) to (B, D), e.g. a ResNet-18 with fc = nn.Identity()) over
    the images in idx (default: all 27,000) and return a (27000, D) tensor with those rows filled. Pass it to
    train(..., feats=...) to train a head on fixed features without rerunning the backbone every epoch."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    X = load_eurosat(data_dir)[0]
    idx = torch.arange(len(X)) if idx is None else idx
    backbone = backbone.to(device).eval(); out = None
    for i in range(0, len(idx), 250):
        b = idx[i:i + 250]; f = backbone(prep(X[b], size).to(device)).cpu()
        if out is None:
            out = torch.zeros(len(X), f.shape[1])
        out[b] = f
    return out

def train(config=None, model=None, feats=None, verbose=True):
    cfg = {**DEFAULTS, **(config or {})}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown hyperparameters: {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    if model is not None:
        cfg["method"] = "custom"
    if cfg["method"] not in METHOD_DEFAULTS:
        raise ValueError(f"method must be one of {sorted(METHOD_DEFAULTS)}")
    d_epochs, d_lr, d_wd, d_bs = METHOD_DEFAULTS[cfg["method"]]
    epochs = cfg["epochs"] or d_epochs; lr = cfg["lr"] or d_lr; wd = cfg["weight_decay"] or d_wd; bs = cfg["batch_size"] or d_bs
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    X, Y, classes, split = load_eurosat(cfg["data_dir"])
    pool = split["pool"]
    if cfg["n"]:
        per_class = cfg["n"] // 10
        train_idx = torch.cat([pool[Y[pool] == c][:per_class] for c in range(10)])
    else:
        train_idx = pool
    val_idx, test_idx = split["val"], split["test"]
    method, size = cfg["method"], cfg["size"]
    if model is not None:
        pass                                          # the challenge path: the model and (optionally) feats are yours
    elif method == "frozen":
        need = torch.cat([train_idx, val_idx] + ([test_idx] if cfg["test"] else []))
        feats = frozen_features(need, size, device, cfg["data_dir"], verbose)
        model = nn.Linear(512, 10)
    elif method == "finetune":
        model = pretrained_resnet18()
    elif method == "scratch":
        from torchvision.models import resnet18
        model = resnet18(num_classes=10)
    else:
        size = 64
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
    model = model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    inputs = lambda b: feats[b].to(device) if feats is not None else prep(X[b], size).to(device)
    if verbose:
        print(f"EuroSAT {method}: {len(train_idx)} training images, {n_params:,} trainable parameters, {epochs} epochs, "
              f"AdamW lr {lr:g}, batch {bs}, input {size}x{size}; device {device}", flush=True)

    def evaluate(idx):
        model.eval(); loss = correct = 0
        with torch.no_grad():
            for i in range(0, len(idx), 500):
                b = idx[i:i + 500]; out = model(inputs(b)); y = Y[b].to(device)
                loss += F.cross_entropy(out, y, reduction="sum").item(); correct += (out.argmax(1) == y).sum().item()
        return loss / len(idx), correct / len(idx)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds")}
    best = (math.inf, None, 0); diverged = False
    for epoch in range(1, epochs + 1):
        model.train(); t0 = time.time(); run_loss = run_correct = 0
        order = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(order), bs):
            b = order[i:i + bs]; x = inputs(b); y = Y[b].to(device)
            if cfg["augment"] and feats is None and torch.rand(()) < 0.5:
                x = x.flip(3)
            optimizer.zero_grad(); logits = model(x); loss = F.cross_entropy(logits, y); loss.backward(); optimizer.step()
            run_loss += loss.item() * len(b); run_correct += (logits.argmax(1) == y).sum().item()
        val_loss, val_acc = evaluate(val_idx)
        row = (run_loss / len(train_idx), run_correct / len(train_idx), val_loss, val_acc, lr, time.time() - t0)
        for k, v in zip(history, row):
            history[k].append(v)
        if verbose:
            print(f"epoch {epoch:3d}   train loss {row[0]:.3f} acc {row[1]:.3f}   val loss {val_loss:.3f} acc {val_acc:.3f}   {row[5]:.1f}s", flush=True)
        if not math.isfinite(row[0]):
            diverged = True; break
        if val_loss < best[0]:
            best = (val_loss, copy.deepcopy(model.state_dict()), epoch)

    secs = sorted(history["seconds"])
    summary = {
        "n_params": n_params, "n_train": len(train_idx), "best_epoch": best[2],
        "val_loss_at_best": best[0] if best[2] else None,
        "val_acc_at_best": history["val_acc"][best[2] - 1] if best[2] else None,
        "max_val_acc": max(history["val_acc"]) if history["val_acc"] else None,
        "final_train_loss": history["train_loss"][-1], "final_train_acc": history["train_acc"][-1],
        "final_val_loss": history["val_loss"][-1], "final_val_acc": history["val_acc"][-1],
        "sec_per_epoch": secs[len(secs) // 2], "diverged": diverged, "test_acc": None,
    }
    if best[2]:
        model.load_state_dict(best[1])
    if cfg["test"] and best[2]:
        summary["test_acc"] = evaluate(test_idx)[1]
        if verbose:
            print(f"test accuracy (epoch-{best[2]} model): {summary['test_acc']:.3f}", flush=True)
    model = model.cpu().eval()
    clean = lambda v: v if not isinstance(v, float) or math.isfinite(v) else None
    history = {k: [clean(v) for v in vs] for k, vs in history.items()}
    summary = {k: clean(v) for k, v in summary.items()}
    return {"config": cfg, "history": history, "summary": summary, "model": model}

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--show-defaults":
        print(json.dumps(DEFAULTS, indent=1))
    elif len(sys.argv) == 2 and sys.argv[1] == "--download":
        X, Y, classes, _ = load_eurosat(); pretrained_resnet18()
        print(f"EuroSAT: {len(Y)} images, classes {classes}; ResNet-18 weights cached. Ready.")
    elif len(sys.argv) == 3:
        result = train(json.load(open(sys.argv[1])))
        result.pop("model")
        json.dump(result, open(sys.argv[2], "w"), indent=1)
        print(f"result written to {sys.argv[2]}")
    else:
        print(__doc__)
