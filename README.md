# The labs of weeks 1–2: an MLP on CIFAR-10, then a CNN

AI Methods for Science (77222). The tasks: `lab_mlp.pdf` (Part A the first task; Part B E0–E11 with `mlp_lab.py`; Part C E12–E15 with `mlp_lab2.py`) and `lab_cnn.pdf` (E16–E19 with `cnn_lab.py`). What is done when is on the last page of `lab_mlp.pdf`.

Setup: any Python >= 3.11 with the packages in `requirements.txt` (torch, torchvision, numpy, pandas, matplotlib). Use whatever you normally use, e.g.

```
pip install -r requirements.txt     # in a venv or a conda env
uv run mlp_lab.py --download        # or: uv installs them itself (pyproject.toml)
```

Then, once, before class:

```
python mlp_lab.py --download        # fetches CIFAR-10 (170 MB) into ./data; can take half an hour
python mlp_lab.py --show-defaults   # the hyperparameters
```

From Python (inside this folder):

```python
from mlp_lab import train
result = train({"lr": 0.1, "epochs": 10})
result["summary"]      # n_params, best_epoch, val_acc_at_best, final_train_acc, sec_per_epoch, ...
result["history"]      # per epoch: train_loss, train_acc, val_loss, val_acc, seconds
```

From the shell: `python mlp_lab.py config.json result.json`.

Do not edit `mlp_lab.py`. Build your experiment pipeline around it, with whatever AI assistant you like.

## The other two scripts

`mlp_lab2.py`: `mlp_lab.py` with new hyperparameters (`optimizer` sgd/adam/adamw, `momentum`, `nesterov`, `weight_decay`, `schedule` none/cosine/step, `warmup_epochs`, `init`, `batchnorm`, `augment`). Same defaults, so `train({})` reproduces E1 exactly; same result format plus an `lr` entry in `history`. Your E0 pipeline works unchanged after `from mlp_lab2 import train`.

`cnn_lab.py` (imports `mlp_lab2.py`): same `train(config)` and result format; the ansatz is a small CNN (`channels`, `convs_per_block`, `kernel`, `batchnorm`; `channels=[]` is the MLP) plus `shuffle_pixels` for E16. Defaults are the E15 recipe (AdamW, cosine, augmentation) and `device="auto"` (a GPU or an Apple GPU when there is one; on a CPU, run 5 epochs instead of 10).

```python
from mlp_lab2 import train
train({"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.05, "augment": True, "schedule": "cosine", "epochs": 30})
from cnn_lab import train
train({"epochs": 10})                        # the default CNN, ~95k parameters
```

Do not edit the scripts.

## Submitting the report

One issue per group: open [New issue](../../issues/new/choose), choose **Lab report**, attach your report and fill in the form. Submitted reports are public; that is intended: at the session they are due we put them next to each other and next to our reference curves.
