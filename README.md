# Session 2 lab: an MLP on CIFAR-10

AI Methods for Science (77222). The tasks are in `lab_session2.pdf`.

```
uv run mlp_lab.py --download        # once, before class: installs PyTorch, fetches CIFAR-10 (170 MB)
uv run mlp_lab.py --show-defaults   # the hyperparameters
```

From Python (inside this folder, `uv run python` or `uv run your_script.py`):

```python
from mlp_lab import train
result = train({"lr": 0.1, "epochs": 10})
result["summary"]      # n_params, best_epoch, val_acc_at_best, final_train_acc, sec_per_epoch, ...
result["history"]      # per epoch: train_loss, train_acc, val_loss, val_acc, seconds
```

From the shell: `uv run mlp_lab.py config.json result.json`.

Do not edit `mlp_lab.py`. Build your experiment pipeline around it, with whatever AI assistant you like.
