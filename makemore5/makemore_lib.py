from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

class Layer:
    """Interface representing one layer in the neural network"""

    training: bool
    out: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Used to forward the neural network"""
        ... 
    
    def parameters(self) -> list[torch.Tensor]:
        """Return a list of trainable parameters"""
        ...

class Container:
    """Interface representing a container for layer of the neural network"""

    layers: list[Layer]

    def __call__(self, x_input: torch.Tensor) -> torch.Tensor:
        """Forwards the `x_input` trough the network"""
        ...

    def parameters(self) -> list[torch.Tensor]:
        """Return all the trainable parameters"""
        ...

class Linear(Layer):
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    out: torch.Tensor
    training: bool 

    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        self.weight = torch.randn((fan_in), (fan_out), generator=generator) / (fan_in**0.5)
        self.bias = torch.zeros((fan_out)) if bias else None
        self.training = True

    def __call__(self, x: torch.Tensor):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d(Layer):
    dim: int
    eps: float
    momentum: float
    training: bool
    gamma: torch.Tensor
    beta: torch.Tensor
    running_mean: torch.Tensor
    running_var: torch.Tensor
    out: torch.Tensor

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

        self.training = True
    
    def __call__(self, x: torch.Tensor):
        assert x.ndim <= 3, "Batch Norm layer supports only 3d tensors"

        if self.training:
            avg_dims = tuple(range(x.ndim))
            x_mean = x.mean(avg_dims, keepdim=True)
            x_var = x.std(avg_dims, keepdim=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps) 
        self.out = self.gamma * x_hat + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh(Layer):
    training: bool
    out: torch.Tensor

    def __init__(self):
        self.training = True

    def __call__(self, x: torch.Tensor):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding(Layer):
    weight: torch.Tensor 
    out: torch.Tensor

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, indices: torch.Tensor):
        self.out = self.weight[indices]
        return self.out
    
    def parameters(self):
        return [self.weight]


class FlattenConsecutive(Layer):
    n_pairs: int
    out: torch.Tensor

    def __init__(self, n_pairs: int):
        self.n_pairs = n_pairs

    def __call__(self, x: torch.Tensor):
        b, t, c = x.shape
        x = x.view(b, t//self.n_pairs, c*self.n_pairs)
        if x.shape[1] == 1:
            x = x.squeeze()
        self.out = x 
        return self.out

    def parameters(self):
        return []


class Sequential(Container):
    layers: list[Layer]
    out: torch.Tensor

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def __call__(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def build_dataset(words: list[str], block_size: int, stoi: dict[str, int]):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for c in w + ".":
            idx = stoi[c]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    print("X:", X.shape, X.dtype)
    print("Y:", Y.shape, Y.dtype)
    return X, Y


def step_decay(epoch: int, initial_lrate=0.1, drop=0.5, epochs_drop=100):
   return initial_lrate * np.pow(drop, np.floor((1+epoch)/epochs_drop))


@torch.no_grad()
def split_loss_eval(
        split_name: str, 
        X_set: torch.Tensor, 
        Y_set: torch.Tensor, 
        container: Container
):
    x = X_set.clone().detach()
    for layer in container.layers:
        x = layer(x)
    loss = F.cross_entropy(x, Y_set)
    print(f"Split '{split_name}' has loss of {loss.item()}")
    return loss

