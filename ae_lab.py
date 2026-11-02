"""AI Methods for Science (77222), session 7 lab: an autoencoder on CIFAR-10. Same loop, no labels: y = x.

Same interface as cnn_lab.py: train(config) -> {"config", "history", "summary", "model"}. The network is a
convolutional encoder (stride-2 convs 32 -> 16 -> 8 -> 4, then a linear layer to `latent` numbers) and the
mirror-image decoder (linear, then transposed convs back to 3 x 32 x 32). The loss is the mean squared error
between the input image and its reconstruction, pixels in [0, 1].

    from ae_lab import train, pca_mse, probe, anomaly_scores, auroc
    r = train({"latent": 32, "epochs": 10})
    r["summary"]["val_mse"], r["summary"]["n_params"]
    pca_mse(32)                           # the same number for PCA with 32 components (no training)
    probe(r)                              # a linear classifier on the 32 codes: validation accuracy
    s = anomaly_scores(r)                 # reconstruction error of the validation set and of five "anomalous" sets
    auroc(s["val"], s["noise"])

    python ae_lab.py --show-defaults
    python ae_lab.py config.json result.json

Hyperparameters (all optional):
    latent          size of the bottleneck d (32). 3072 numbers go in, d come out of the encoder.
    channels        channels of the three encoder blocks ([32, 64, 128]); the decoder mirrors them
    noise           std of Gaussian noise added to the *input* during training; the target stays clean (0 = plain
                    autoencoder). For E25 (denoising).
    exclude_class   a CIFAR-10 class (0-9) removed from the training and validation sets, or -1 for none. For E24:
                    its images are then "anomalies" the network never saw.
    optimizer, lr, momentum, weight_decay, schedule, warmup_epochs, batch_size, epochs, seed, device, data_dir
                    as in mlp_lab2.py (defaults: AdamW 1e-3, weight decay 0.05, cosine, 10 epochs)

Metrics returned: summary = n_params, val_mse (best epoch), train_mse, best_epoch, sec_per_epoch, device, excluded;
history = per epoch train_mse, val_mse, lr, seconds. result["model"] is the network at the best epoch, on the CPU.

Helpers (all use the validation split of the standard 45,000 / 5,000 split, pixels in [0, 1]):
    reconstruct(result, x)     -> x_hat, same shape as x
    encode(result, x)          -> the codes, (N, latent)
    score(result, x)           -> per-image reconstruction error, mean over the 3072 pixels, (N,)
    pca_mse(d)                 -> validation MSE of PCA with d components fitted on the training images
    pca_reconstruct(x, d)      -> PCA reconstruction with d components
    probe(result)              -> validation accuracy of a linear classifier trained on the codes (labels used here only)
    probe_features(f_tr, y_tr, f_va, y_va)  the same for any features, e.g. pixels or PCA scores
    anomaly_sets(result)       -> {"val", "excluded", "permuted", "noise", "constant", "eurosat"}: the images
    anomaly_scores(result)     -> the same keys: reconstruction errors
    auroc(normal, anomalous)   -> P(score of an anomaly > score of a normal image)
"""
import copy, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_lab2 import load_cifar10, lr_factor, make_optimizer

DEFAULTS = {
    "latent": 32, "channels": [32, 64, 128], "noise": 0.0, "exclude_class": -1,
    "optimizer": "adamw", "lr": 1e-3, "momentum": 0.0, "nesterov": False, "weight_decay": 0.05,
    "schedule": "cosine", "warmup_epochs": 0, "batch_size": 128, "epochs": 10,
    "seed": 0, "device": "auto", "data_dir": "./data",
}
CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def pick_device(device="auto"):
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ---- data: pixels in [0, 1] ----------------------------------------------------------
_data = {}
def load_pixels(data_dir="./data"):
    """(x_train, y_train, x_val, y_val) as tensors in [0, 1], shapes (N, 3, 32, 32); the standard 45,000 / 5,000 split."""
    if data_dir not in _data:
        tr, va, _ = load_cifar10(data_dir, normalize=False)      # values 0..255
        def unpack(subset):
            idx = torch.tensor(subset.indices)
            return subset.dataset.tensors[0][idx] / 255, subset.dataset.tensors[1][idx]
        _data[data_dir] = (*unpack(tr), *unpack(va))
    return _data[data_dir]

# ---- model --------------------------------------------------------------------------
class ConvAE(nn.Module):
    def __init__(self, latent=32, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        def down(a, b):                                           # halves the grid: conv 4x4, stride 2
            return nn.Sequential(nn.Conv2d(a, b, 4, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU())
        def up(a, b):                                             # doubles the grid: transposed conv 4x4, stride 2
            return nn.Sequential(nn.ConvTranspose2d(a, b, 4, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU())
        self.c3 = c3
        self.encoder = nn.Sequential(down(3, c1), down(c1, c2), down(c2, c3), nn.Flatten(),   # 32 -> 16 -> 8 -> 4
                                     nn.Linear(c3 * 16, latent))                              # the bottleneck
        self.decoder = nn.Sequential(nn.Linear(latent, c3 * 16), nn.Unflatten(1, (c3, 4, 4)), nn.BatchNorm2d(c3), nn.ReLU(),
                                     up(c3, c2), up(c2, c1),                                   # 4 -> 8 -> 16
                                     nn.ConvTranspose2d(c1, 3, 4, stride=2, padding=1))        # 16 -> 32, linear output

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ---- training -----------------------------------------------------------------------
def _loader(x, batch_size, shuffle):
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=shuffle)

@torch.no_grad()
def _mse(model, x, device, batch_size=500):
    model.eval()
    out = []
    for (xb,) in _loader(x, batch_size, False):
        xb = xb.to(device)
        out.append(((model(xb) - xb) ** 2).mean(dim=(1, 2, 3)).cpu())
    return torch.cat(out)

def train(config=None, verbose=True):
    cfg = {**DEFAULTS, **(config or {})}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown hyperparameters: {sorted(unknown)}")
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    x_tr, y_tr, x_va, y_va = load_pixels(cfg["data_dir"])
    if cfg["exclude_class"] >= 0:
        x_tr, x_va = x_tr[y_tr != cfg["exclude_class"]], x_va[y_va != cfg["exclude_class"]]
    model = ConvAE(cfg["latent"], cfg["channels"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, cfg)
    loader = _loader(x_tr, cfg["batch_size"], True)
    total_steps, step = cfg["epochs"] * len(loader), 0
    warmup_steps = cfg["warmup_epochs"] * len(loader)
    if verbose:
        print(f"ConvAE latent {cfg['latent']}, channels {cfg['channels']}, {n_params:,} parameters; "
              f"{len(x_tr):,} training images; device {device}", flush=True)
    history = {k: [] for k in ("train_mse", "val_mse", "lr", "seconds")}
    best = {"val_mse": float("inf")}
    for epoch in range(cfg["epochs"]):
        model.train(); t0 = time.time(); running, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            inp = xb + cfg["noise"] * torch.randn_like(xb) if cfg["noise"] > 0 else xb
            lr = cfg["lr"] * lr_factor(step, total_steps, warmup_steps, cfg["schedule"])
            for g in opt.param_groups:
                g["lr"] = lr
            loss = F.mse_loss(model(inp), xb)                     # the target is always the clean image
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            running += loss.item() * len(xb); n += len(xb)
        val_mse = _mse(model, x_va, device).mean().item()
        history["train_mse"].append(running / n); history["val_mse"].append(val_mse)
        history["lr"].append(lr); history["seconds"].append(time.time() - t0)
        if val_mse < best["val_mse"]:
            best = {"val_mse": val_mse, "epoch": epoch + 1, "state": copy.deepcopy(model.state_dict())}
        if verbose:
            print(f"epoch {epoch + 1:3d}  train mse {running / n:.5f}  val mse {val_mse:.5f}  "
                  f"lr {lr:.2e}  {time.time() - t0:.1f}s", flush=True)
    model.load_state_dict(best["state"]); model = model.cpu().eval()
    summary = {"n_params": n_params, "val_mse": best["val_mse"], "best_epoch": best["epoch"],
               "train_mse": history["train_mse"][-1], "sec_per_epoch": float(np.mean(history["seconds"])),
               "device": device, "excluded": cfg["exclude_class"]}
    return {"config": cfg, "history": history, "summary": summary, "model": model}

# ---- helpers ------------------------------------------------------------------------
@torch.no_grad()
def _apply(fn, x, batch_size=500):
    return torch.cat([fn(xb).cpu() for (xb,) in _loader(x, batch_size, False)])

def _device_model(result):
    device = pick_device(result["config"]["device"])
    return result["model"].to(device).eval(), device

def reconstruct(result, x):
    model, device = _device_model(result)
    out = _apply(lambda xb: model(xb.to(device)), x)
    result["model"].cpu()
    return out

def encode(result, x):
    model, device = _device_model(result)
    out = _apply(lambda xb: model.encoder(xb.to(device)), x)
    result["model"].cpu()
    return out

def score(result, x):
    """Reconstruction error per image: the anomaly score."""
    return ((reconstruct(result, x) - x) ** 2).mean(dim=(1, 2, 3)).numpy()

_pca = {}
def _pca_basis(data_dir="./data"):
    """Mean and eigenvectors (columns, descending variance) of the 3072 x 3072 covariance of the training images."""
    if data_dir not in _pca:
        x_tr = load_pixels(data_dir)[0].reshape(-1, 3072).double()
        mean = x_tr.mean(0)
        cov = (x_tr - mean).T @ (x_tr - mean) / len(x_tr)
        evals, evecs = torch.linalg.eigh(cov)
        _pca[data_dir] = (mean.float(), evecs.flip(1).float(), evals.flip(0).float())
    return _pca[data_dir]

def pca_reconstruct(x, d, data_dir="./data"):
    mean, V, _ = _pca_basis(data_dir)
    flat = x.reshape(len(x), -1) - mean
    return ((flat @ V[:, :d]) @ V[:, :d].T + mean).reshape(x.shape)

def pca_scores(x, d, data_dir="./data"):
    mean, V, _ = _pca_basis(data_dir)
    return (x.reshape(len(x), -1) - mean) @ V[:, :d]

def pca_mse(d, data_dir="./data"):
    """Validation MSE of PCA with d components: the linear autoencoder's optimum, by eigendecomposition, no training."""
    x_va = load_pixels(data_dir)[2]
    return ((pca_reconstruct(x_va, d, data_dir) - x_va) ** 2).mean().item()

def probe_features(f_tr, y_tr, f_va, y_va, steps=300, device="auto"):
    """Multinomial logistic regression on standardized features (full-batch Adam, weight decay 1e-4): validation accuracy."""
    device = pick_device(device)
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-6
    f_tr, f_va = ((f_tr - mu) / sd).to(device), ((f_va - mu) / sd).to(device)
    y_tr, y_va = y_tr.to(device), y_va.to(device)
    torch.manual_seed(0)
    head = nn.Linear(f_tr.shape[1], int(y_tr.max()) + 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        loss = F.cross_entropy(head(f_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (head(f_va).argmax(1) == y_va).float().mean().item()

def probe(result, steps=300):
    """What is in the code? A linear classifier on the encoder's output (the first time labels are used)."""
    x_tr, y_tr, x_va, y_va = load_pixels(result["config"]["data_dir"])
    return probe_features(encode(result, x_tr), y_tr, encode(result, x_va), y_va, steps, result["config"]["device"])

def anomaly_sets(result, n=1000, seed=0):
    """The validation images and five sets the network never trained on, as tensors: "val", "excluded" (if a class was
    excluded), "permuted" (validation images under one pixel permutation), "noise" (uniform), "constant" (one colour per
    image), "eurosat" (satellite patches resized to 32 x 32, if E21's data are in data_dir)."""
    cfg = result["config"]
    x_tr, y_tr, x_va, y_va = load_pixels(cfg["data_dir"])
    g = torch.Generator().manual_seed(seed)
    sets = {}
    if cfg["exclude_class"] >= 0:
        sets["val"] = x_va[y_va != cfg["exclude_class"]]
        sets["excluded"] = x_tr[y_tr == cfg["exclude_class"]][:n]           # never seen in training
    else:
        sets["val"] = x_va
    perm = torch.randperm(3072, generator=g)
    sets["permuted"] = x_va[:n].reshape(n, -1)[:, perm].reshape(n, 3, 32, 32)   # same pixels, no structure
    sets["noise"] = torch.rand(n, 3, 32, 32, generator=g)                        # uniform noise
    sets["constant"] = torch.rand(n, 3, 1, 1, generator=g).expand(n, 3, 32, 32).clone()   # one colour per image
    cache = os.path.join(cfg["data_dir"], "eurosat_uint8.pt")
    if os.path.exists(cache):                                                     # satellite patches, if E21 was run
        X = torch.load(cache)[0]
        idx = torch.randperm(len(X), generator=g)[:n]
        sets["eurosat"] = F.interpolate(X[idx].float() / 255, size=32, mode="area")
    return sets

def anomaly_scores(result, n=1000, seed=0):
    """Reconstruction errors of the validation images and of the five sets of anomaly_sets."""
    return {k: score(result, v) for k, v in anomaly_sets(result, n, seed).items()}

def auroc(normal, anomalous):
    """Probability that a random anomaly scores above a random normal image (the area under the ROC curve)."""
    normal, anomalous = np.asarray(normal), np.asarray(anomalous)
    order = np.argsort(np.concatenate([normal, anomalous]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    r_anom = ranks[len(normal):].sum()
    return (r_anom - len(anomalous) * (len(anomalous) + 1) / 2) / (len(normal) * len(anomalous))

def roc_curve(normal, anomalous):
    """False-alarm rate and detection rate for every threshold, for a plot."""
    s = np.concatenate([normal, anomalous]); lab = np.concatenate([np.zeros(len(normal)), np.ones(len(anomalous))])
    order = np.argsort(-s); lab = lab[order]
    tpr = np.cumsum(lab) / lab.sum(); fpr = np.cumsum(1 - lab) / (1 - lab).sum()
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])

# ---- command line -------------------------------------------------------------------
if __name__ == "__main__":
    if "--show-defaults" in sys.argv:
        print(json.dumps(DEFAULTS, indent=2)); sys.exit()
    cfg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    result = train(cfg)
    print(json.dumps(result["summary"], indent=2))
    if len(sys.argv) > 2:
        json.dump({k: result[k] for k in ("config", "history", "summary")}, open(sys.argv[2], "w"), indent=2)
