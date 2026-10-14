# Session 2 lab: an MLP on CIFAR-10

AI Methods for Science (77222). The tasks are in `lab_session2.pdf`.

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
