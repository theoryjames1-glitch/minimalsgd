# minimalsgd

Here are two clean options you can use right away: a framework-agnostic SGD (NumPy-friendly) and a drop-in PyTorch optimizer.

# 1) Minimal, framework-agnostic SGD (with momentum, nesterov, weight decay, grad clipping)

```python
from typing import Dict, Any, Iterable, Optional
import math

class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Params format:
      - params: dict[str, Dict[str, Any]]
          Each entry maps a name -> {"value": ndarray-like, "grad": ndarray-like}
          Only 'value' is required at construction; set 'grad' before calling step().
    """
    def __init__(
        self,
        params: Dict[str, Dict[str, Any]],
        lr: float = 1e-2,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        clip_grad_norm: Optional[float] = None,
    ):
        assert lr > 0
        assert momentum >= 0
        if nesterov:
            assert momentum > 0, "Nesterov requires momentum > 0"
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self._velocity = {k: None for k in params.keys()}

    def zero_grad(self):
        for p in self.params.values():
            if "grad" in p and p["grad"] is not None:
                p["grad"][...] = 0

    def _flatten_grads(self) -> Iterable:
        for name, p in self.params.items():
            g = p.get("grad", None)
            if g is not None:
                yield g

    def _global_grad_norm(self):
        # Works for NumPy arrays or array-likes with ** operator and sum/reshape
        sq_sum = 0.0
        for g in self._flatten_grads():
            sq_sum += float((g * g).sum())
        return math.sqrt(sq_sum)

    def step(self):
        # Optional global grad clipping
        if self.clip_grad_norm is not None:
            total_norm = self._global_grad_norm()
            if total_norm > 0 and total_norm > self.clip_grad_norm:
                scale = self.clip_grad_norm / (total_norm + 1e-12)
                for p in self.params.values():
                    g = p.get("grad", None)
                    if g is not None:
                        p["grad"][...] = g * scale

        # Parameter update
        for name, p in self.params.items():
            w = p["value"]
            g = p.get("grad", None)
            if g is None:
                continue

            if self.weight_decay != 0.0:
                g = g + self.weight_decay * w

            v = self._velocity[name]
            if self.momentum > 0.0:
                if v is None:
                    # Initialize velocity as zeros of same shape
                    v = 0.0 * w
                v = self.momentum * v + g
                self._velocity[name] = v
                if self.nesterov:
                    update = g + self.momentum * v
                else:
                    update = v
            else:
                update = g

            p["value"][...] = w - self.lr * update
```

Quick example (NumPy linear regression):

```python
import numpy as np

# Fake data: y = 3x + 2 + noise
np.random.seed(0)
X = np.random.randn(200, 1)
y = 3 * X[:, 0] + 2 + 0.1 * np.random.randn(200)

# Params
W = np.zeros((1, 1))
b = np.zeros((1,))
params = {
    "W": {"value": W, "grad": np.zeros_like(W)},
    "b": {"value": b, "grad": np.zeros_like(b)},
}

opt = SGD(params, lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0)

for step in range(300):
    # forward
    yhat = (X @ params["W"]["value"]).reshape(-1) + params["b"]["value"][0]
    loss = ((yhat - y) ** 2).mean()

    # grads
    grad_yhat = 2 * (yhat - y) / y.size
    params["W"]["grad"][...] = (X.T @ grad_yhat.reshape(-1,1))
    params["b"]["grad"][...] = np.array([grad_yhat.sum()])

    opt.step()
    opt.zero_grad()

# Learned params (close to 3 and 2)
print("W:", params["W"]["value"].ravel()[0], "b:", params["b"]["value"][0])
```

# 2) PyTorch: custom SGD optimizer (momentum + Nesterov + weight decay)

> PyTorch already has `torch.optim.SGD`, but here’s a from-scratch version for learning/debugging.

```python
import torch
from torch.optim import Optimizer

class MySGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, nesterov=False, weight_decay=0.0):
        if lr <= 0: raise ValueError("Invalid lr")
        if momentum < 0: raise ValueError("Invalid momentum")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov requires momentum > 0")

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)  # weight decay

                state = self.state[p]
                if mom != 0:
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(d_p)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=mom)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)
        return loss
```

Usage:

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 1))
opt = MySGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)
loss_fn = nn.MSELoss()

x = torch.randn(32, 10)
y = torch.randn(32, 1)

for _ in range(200):
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()

print("Final loss:", float(loss))
```

Want this adapted for a specific framework (JAX, TensorFlow, Keras optimizer API) or with extras like per-param groups, cosine decay, or AMSGrad-style buffers? I can wire that up.
