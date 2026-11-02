"""AI Methods for Science (77222), session 8 lab: a variational autoencoder on CIFAR-10. Monday's network with a
distribution in the bottleneck and a bound on log p(x) as the loss.

Same interface as ae_lab.py: train(config) -> {"config", "history", "summary", "model"}. The encoder outputs a mean and a
log-variance per latent coordinate, z = mu + sigma * eps is sampled (reparameterization), the decoder is Monday's. The
loss per image is the negative ELBO with a Gaussian likelihood p(x|z) = N(x_hat, sigma^2) per pixel:

    loss = sum_pixels (x - x_hat)^2 / (2 sigma^2)  +  beta * KL( N(mu, s^2) || N(0, 1) ),   KL = 1/2 sum_j (mu_j^2 + s_j^2 - log s_j^2 - 1)

With beta = 1 this is -ELBO up to the constant 1536 ln(2 pi sigma^2); score(kind="elbo") adds that constant and the
3072 ln 255 that turns the density into the probability of the 8-bit image, so it is an upper bound on -log p(x) in nats
and bits_per_dim() compares with PNG. Changing beta at fixed sigma is the same as changing sigma^2 -> sigma^2 / beta:
the "beta-VAE" is a Gaussian likelihood with a different noise level. Pixels in [0, 1].

    from vae_lab import train, sample, interpolate, score, complexity, anomaly_scores, auroc
    r = train({"latent": 128, "epochs": 10})
    r["summary"]["val_mse"], r["summary"]["val_kl"], r["summary"]["val_neg_elbo"]
    sample(r, 16)                                 # 16 images decoded from z ~ N(0, I)
    bits_per_dim(r)                               # the bound on -log2 p(x) per pixel value, validation set
    s = anomaly_scores(r, kind="elbo")            # -ELBO of the validation set and of the five anomalous sets
    s = anomaly_scores(r, kind="elbo-complexity") # the same minus the PNG-compressed length in nats (Serra et al. 2020)

Hyperparameters: latent, channels, exclude_class, optimizer, lr, weight_decay, schedule, warmup_epochs, batch_size, epochs,
seed, device, data_dir as in ae_lab.py (defaults AdamW 1e-3, weight decay 0.05, cosine, 10 epochs), plus
    sigma           the likelihood's noise level per pixel (0.1): how far a reconstruction may be from the image before
                    it costs more than the code
    beta            weight of the KL term (1.0). beta < 1: sharper reconstructions, a code that ignores the prior;
                    beta > 1: blurrier, a code closer to the prior (the "beta-VAE")

Metrics: summary = n_params, val_loss (best epoch), val_mse (per pixel, decoding the mean), val_kl (nats per image),
val_neg_elbo (nats per 8-bit image, constants included), val_bits_per_dim, best_epoch, sec_per_epoch, device, excluded,
sigma, beta;
history per epoch: train_loss, val_loss, val_mse, val_kl, lr, seconds.

Helpers (validation split of the 45,000 / 5,000 split, pixels in [0, 1]):
    encode(result, x)          -> (mu, logvar), each (N, latent)
    reconstruct(result, x)     -> x_hat decoded from mu (no sampling)
    sample(result, n, seed=0)  -> n images decoded from z ~ N(0, I)
    interpolate(result, x1, x2, steps=8)  -> images decoded along the line from mu(x1) to mu(x2)
    score(result, x, kind)     -> per-image anomaly score: "recon" (sum of squared errors from mu), "kl", "elbo" (the bound
                                  on -log p(x) in nats, constants included), "elbo-complexity" (elbo - PNG length in nats)
    complexity(x)              -> PNG-compressed length of each image, in bits
    bits_per_dim(result, x=None) -> mean of elbo / (3072 ln 2) over x (default: the validation set)
    anomaly_sets, anomaly_scores(result, kind), auroc, roc_curve, probe_features   as in ae_lab.py
"""
import copy, io, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_lab2 import lr_factor, make_optimizer
import ae_lab
from ae_lab import CLASSES, _apply, _loader, anomaly_sets, auroc, load_pixels, pick_device, probe_features, roc_curve

DEFAULTS = {k: v for k, v in ae_lab.DEFAULTS.items() if k != "noise"}
DEFAULTS["sigma"] = 0.1
DEFAULTS["beta"] = 1.0

def _const(sigma):
    """Constant that turns sum (x - x_hat)^2 / (2 sigma^2) + KL into a bound on -log p of the 8-bit image, nats."""
    return 1536 * math.log(2 * math.pi * sigma ** 2) + 3072 * math.log(255)

class ConvVAE(nn.Module):
    def __init__(self, latent=32, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        def down(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 4, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU())
        def up(a, b):
            return nn.Sequential(nn.ConvTranspose2d(a, b, 4, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU())
        self.latent = latent
        self.trunk = nn.Sequential(down(3, c1), down(c1, c2), down(c2, c3), nn.Flatten())           # 32 -> 4, c3*16 numbers
        self.head = nn.Linear(c3 * 16, 2 * latent)                                                 # mean and log-variance
        self.decoder = nn.Sequential(nn.Linear(latent, c3 * 16), nn.Unflatten(1, (c3, 4, 4)), nn.BatchNorm2d(c3), nn.ReLU(),
                                     up(c3, c2), up(c2, c1), nn.ConvTranspose2d(c1, 3, 4, stride=2, padding=1))

    def encode(self, x):
        h = self.head(self.trunk(x))
        return h[:, :self.latent], h[:, self.latent:]                                              # mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)                                    # reparameterization
        return self.decoder(z), mu, logvar

def _terms(model, x, sample=True):
    """Per-image sum of squared errors and KL to the prior."""
    mu, logvar = model.encode(x)
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
    recon = ((model.decoder(z) - x) ** 2).sum(dim=(1, 2, 3))
    kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(dim=1)
    return recon, kl

@torch.no_grad()
def _evaluate(model, x, cfg, device, batch_size=500):
    model.eval(); out = {"recon": [], "kl": []}
    for (xb,) in _loader(x, batch_size, False):
        r, k = _terms(model, xb.to(device), sample=False)
        out["recon"].append(r.cpu()); out["kl"].append(k.cpu())
    recon, kl = torch.cat(out["recon"]), torch.cat(out["kl"])
    sigma, beta = cfg["sigma"], cfg["beta"]
    neg_elbo = (recon / (2 * sigma ** 2) + kl + _const(sigma)).mean().item()
    return {"val_loss": (recon / (2 * sigma ** 2) + beta * kl).mean().item(), "val_mse": (recon / 3072).mean().item(),
            "val_kl": kl.mean().item(), "val_neg_elbo": neg_elbo, "val_bits_per_dim": neg_elbo / (3072 * math.log(2))}

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
    model = ConvVAE(cfg["latent"], cfg["channels"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, cfg)
    loader = _loader(x_tr, cfg["batch_size"], True)
    total_steps, step = cfg["epochs"] * len(loader), 0
    warmup_steps = cfg["warmup_epochs"] * len(loader)
    if verbose:
        print(f"ConvVAE latent {cfg['latent']}, sigma {cfg['sigma']}, beta {cfg['beta']}, {n_params:,} parameters; {len(x_tr):,} training images; "
              f"device {device}", flush=True)
    history = {k: [] for k in ("train_loss", "val_loss", "val_mse", "val_kl", "lr", "seconds")}
    best = {"val_loss": float("inf")}
    for epoch in range(cfg["epochs"]):
        model.train(); t0 = time.time(); running, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            lr = cfg["lr"] * lr_factor(step, total_steps, warmup_steps, cfg["schedule"])
            for g in opt.param_groups:
                g["lr"] = lr
            recon, kl = _terms(model, xb)
            loss = (recon / (2 * cfg["sigma"] ** 2) + cfg["beta"] * kl).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            running += loss.item() * len(xb); n += len(xb)
        ev = _evaluate(model, x_va, cfg, device)
        history["train_loss"].append(running / n); history["val_loss"].append(ev["val_loss"])
        history["val_mse"].append(ev["val_mse"]); history["val_kl"].append(ev["val_kl"])
        history["lr"].append(lr); history["seconds"].append(time.time() - t0)
        if ev["val_loss"] < best["val_loss"]:
            best = {**ev, "epoch": epoch + 1, "state": copy.deepcopy(model.state_dict())}
        if verbose:
            print(f"epoch {epoch + 1:3d}  train loss {running / n:7.1f}  val loss {ev['val_loss']:7.1f}  "
                  f"val mse {ev['val_mse']:.5f}  val KL {ev['val_kl']:6.1f}  lr {lr:.2e}  {time.time() - t0:.1f}s", flush=True)
    model.load_state_dict(best["state"]); model = model.cpu().eval()
    summary = {"n_params": n_params, "val_loss": best["val_loss"], "val_mse": best["val_mse"], "val_kl": best["val_kl"],
               "val_neg_elbo": best["val_neg_elbo"], "val_bits_per_dim": best["val_bits_per_dim"], "best_epoch": best["epoch"],
               "sec_per_epoch": float(np.mean(history["seconds"])), "device": device,
               "excluded": cfg["exclude_class"], "sigma": cfg["sigma"], "beta": cfg["beta"]}
    return {"config": cfg, "history": history, "summary": summary, "model": model}

# ---- helpers ------------------------------------------------------------------------
def _device_model(result):
    device = pick_device(result["config"]["device"])
    return result["model"].to(device).eval(), device

def encode(result, x):
    model, device = _device_model(result)
    with torch.no_grad():
        out = [model.encode(xb.to(device)) for (xb,) in _loader(x, 500, False)]
    result["model"].cpu()
    return torch.cat([m for m, _ in out]).cpu(), torch.cat([v for _, v in out]).cpu()

def reconstruct(result, x):
    model, device = _device_model(result)
    out = _apply(lambda xb: model.decoder(model.encode(xb.to(device))[0]), x)
    result["model"].cpu()
    return out

def decode(result, z):
    model, device = _device_model(result)
    out = _apply(lambda zb: model.decoder(zb.to(device)), z)
    result["model"].cpu()
    return out

def sample(result, n=16, seed=0):
    """n images decoded from z ~ N(0, I): the generative model's own samples."""
    z = torch.randn(n, result["config"]["latent"], generator=torch.Generator().manual_seed(seed))
    return decode(result, z)

def interpolate(result, x1, x2, steps=8):
    """Images decoded along the straight line from mu(x1) to mu(x2) in latent space."""
    mu, _ = encode(result, torch.stack([x1, x2]))
    t = torch.linspace(0, 1, steps)[:, None]
    return decode(result, (1 - t) * mu[0] + t * mu[1])

def complexity(x):
    """PNG-compressed length of each image in bits: a cheap, model-free measure of how much there is to describe."""
    from PIL import Image
    bits = []
    for img in (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy():
        buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG", optimize=True); bits.append(8 * buf.tell())
    return np.array(bits, dtype=float)

def score(result, x, kind="elbo"):
    """Per-image anomaly score. "recon": sum of squared errors (decoding mu); "kl": KL to the prior; "elbo": the bound on
    -log p(x) of the 8-bit image in nats (recon / 2 sigma^2 + KL + constants); "elbo-complexity": elbo minus the PNG length
    converted to nats (Serra et al. 2020): what the model cannot explain beyond what any compressor finds hard."""
    model, device = _device_model(result)
    rec, kl = [], []
    with torch.no_grad():
        for (xb,) in _loader(x, 500, False):
            r, k = _terms(model, xb.to(device), sample=False); rec.append(r.cpu()); kl.append(k.cpu())
    result["model"].cpu()
    rec, kl = torch.cat(rec).numpy(), torch.cat(kl).numpy()
    if kind == "recon":
        return rec
    if kind == "kl":
        return kl
    sigma = result["config"]["sigma"]
    neg_elbo = rec / (2 * sigma ** 2) + kl + _const(sigma)
    if kind == "elbo":
        return neg_elbo
    if kind == "elbo-complexity":
        return neg_elbo - complexity(x) * math.log(2)
    raise ValueError(f"unknown score kind {kind!r}")

def bits_per_dim(result, x=None):
    """The bound on -log2 p(x) per pixel value, averaged over x (default: the validation set). PNG: about 5.8 on CIFAR-10."""
    if x is None:
        x = load_pixels(result["config"]["data_dir"])[2]
    return float(score(result, x, "elbo").mean() / (3072 * math.log(2)))

def anomaly_scores(result, kind="elbo", n=1000, seed=0):
    """Scores of the validation images and of the five sets of ae_lab.anomaly_sets."""
    return {k: score(result, v, kind) for k, v in anomaly_sets(result, n, seed).items()}

# ---- command line -------------------------------------------------------------------
if __name__ == "__main__":
    if "--show-defaults" in sys.argv:
        print(json.dumps(DEFAULTS, indent=2)); sys.exit()
    cfg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    result = train(cfg)
    print(json.dumps(result["summary"], indent=2))
    if len(sys.argv) > 2:
        json.dump({k: result[k] for k in ("config", "history", "summary")}, open(sys.argv[2], "w"), indent=2)
