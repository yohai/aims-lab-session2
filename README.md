# AI Methods for Science (77222), Part I

This is the repo for Yohai's part of the course: weeks 1–4, deep learning foundations. It has the slides, the lab code and the task sheets, and it is where you hand in your reports. It gets updated after every session, so `git pull` before class.

## Slides

The decks as shown in class are in [`slides/`](slides/), one PDF per lecture, numbered from 0. If a session's deck is missing, it will be there within a day.

## Setup

Any Python 3.11 or newer with the packages in `requirements.txt` (torch, torchvision, numpy, pandas, matplotlib). Use whatever you normally use:

```
pip install -r requirements.txt     # in a venv or a conda env
```

or, with `uv`, nothing to install: `uv run mlp_lab.py --download` fetches the packages itself.

Then, once, before the first lab (it fetches CIFAR-10, 170 MB, into `./data`; can take a while on the campus network):

```
python mlp_lab.py --download
python mlp_lab.py --show-defaults   # the hyperparameters
```

## The labs

The tasks are in `lab_mlp.pdf` (Part A the first task; Part B E0–E11 with `mlp_lab.py`; Part C E12–E15 with `mlp_lab2.py`) and `lab_cnn.pdf` (E16–E20 and E22 with `cnn_lab.py`, E21 with `eurosat_lab.py`). Session 7 (autoencoders): E22–E25 with `ae_lab.py`, the tasks are on the lab frames of the deck until the sheet is out. The last page of `lab_mlp.pdf` says what is done when.

The scripts give you a network and a training loop. You write the experiments around them, with whatever AI assistant you like. From Python, inside this folder:

```python
from mlp_lab import train
result = train({"lr": 0.1, "epochs": 10})
result["summary"]      # n_params, best_epoch, val_acc_at_best, final_train_acc, sec_per_epoch, ...
result["history"]      # per epoch: train_loss, train_acc, val_loss, val_acc, seconds
```

From the shell: `python mlp_lab.py config.json result.json`.

Please do not edit the scripts. Everyone's numbers should be comparable across the room.

**`mlp_lab2.py`** is `mlp_lab.py` with more knobs (`optimizer` sgd/adam/adamw, `momentum`, `nesterov`, `weight_decay`, `schedule` none/cosine/step, `warmup_epochs`, `init`, `batchnorm`, `augment`). Same defaults, so `train({})` reproduces E1 exactly; same result format plus an `lr` entry in `history`. Your E0 pipeline works unchanged after `from mlp_lab2 import train`.

**`cnn_lab.py`** (imports `mlp_lab2.py`): same `train(config)` and result format, but the model is a small CNN (`channels`, `convs_per_block`, `kernel`, `batchnorm`, `skip` for E22; `channels=[]` gives back the MLP), plus `shuffle_pixels` for E16, `classes`, `n_per_class`, `save_weights`, `init_from`, `freeze` and the helper `nearest_mean(weights, classes, n_per_class)` for E20 (transfer within CIFAR-10, Thursday in class; E19 is homework); the result also holds the trained network, `result["model"]`. Defaults are the E15 recipe (AdamW, cosine, augmentation) and `device="auto"`: a GPU or an Apple GPU when there is one. On a CPU, run 5 epochs instead of 10.

**`eurosat_lab.py`** (E21, homework): transfer learning on EuroSAT, 27,000 satellite patches in 10 land-use classes. Same `train(config)` and result format; keys `method` (`mlp`, `scratch`, `frozen`, `finetune`), `n` (labelled training images), `epochs`, `lr`, `size`; or build the model yourself and pass `train(config, model=m)`, with `extract_features(backbone)` for a frozen backbone (the sheet's option B). Run `python eurosat_lab.py --download` once at home (EuroSAT 94 MB, ResNet-18 weights 45 MB). Frozen features are cached in `data/` after the first run.

```python
from mlp_lab2 import train
train({"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.05, "augment": True, "schedule": "cosine", "epochs": 30})

from cnn_lab import train
train({"epochs": 10})                        # the default CNN, ~95k parameters
```

## Handing in a report

One issue per group: open [New issue](../../issues/new/choose), choose **Lab report**, attach the report and fill in the form. Reports are public, on purpose: in the session they are due we put them next to each other and next to our reference runs, and that is where the discussion starts.
