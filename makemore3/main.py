#!/usr/bin/env python


import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class Linear:
    weight: torch.Tensor
    bias: torch.Tensor | None
    out: torch.Tensor
    training: bool 

    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        self.weight = torch.randn((fan_in), (fan_out), generator=generator) / (fan_in**0.5)
        self.bias = torch.zeros((fan_out)) if bias else None
        self.training = False

    def __call__(self, x: torch.Tensor):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
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
        if self.training:
            x_mean = x.mean(0, keepdim=True)
            x_var = x.std(0, keepdim=True)
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


class Tanh:
    out: torch.Tensor
    training: bool

    def __call__(self, x: torch.Tensor):
        self.out = torch.tanh(x)
        self.training = False
        return self.out

    def parameters(self):
        return []


def build_dataset(words: list[str], block_size: int):
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
def split_loss_eval(split_name: str, x_input: torch.Tensor, y_label: torch.Tensor, embed_mat: torch.Tensor):
    embeds = embed_mat[x_input]
    x = embeds.view(embeds.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y_label)
    print(f"Split '{split_name}' has loss of {loss.item()}")
    return loss


if __name__ == "__main__":
    # Seeding Generators
    random.seed(42)
    g = torch.Generator().manual_seed(2147483647)

    # Dataset Preparation
    words = open("./names.txt").read().splitlines()
    chars = ["."] + sorted(list(set("".join(words))))
    stoi = {char: idx for idx, char in enumerate(chars)}
    itos = {idx: char for char, idx in stoi.items()}

    training_percentage = 0.9
    validation_percentage = 0.05
    test_percentage = 0.05

    n1 = int(training_percentage * len(words))
    n2 = int((training_percentage + validation_percentage) * len(words))

    # Context Window
    block_size = 8

    random.shuffle(words) 
    X_train, Y_train = build_dataset(words[:n1], block_size)    # 80%
    X_val, Y_val = build_dataset(words[n1:n2], block_size)      # 10%
    X_test, Y_test = build_dataset(words[n2:], block_size)      # 10%

    # Layers
    vocab_size = len(chars)
    n_embedding = 16
    n_hidden = 128
    
    embed_mat = torch.randn((vocab_size, n_embedding), generator=g)

    layers = [
        Linear(n_embedding * block_size, n_hidden, generator=g, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),

        Linear(n_hidden, n_hidden, generator=g, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),

        Linear(n_hidden, n_hidden, generator=g, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),

        Linear(n_hidden, n_hidden, generator=g, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),

        Linear(n_hidden, n_hidden, generator=g, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),

        Linear(n_hidden, vocab_size, generator=g, bias=False),
        BatchNorm1d(vocab_size),
    ]

    # General Layers Setup
    with torch.no_grad():
        # layers[-1].weight *= 0.1
        layers[-1].gamma *= 0.01
        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= .2
                # layer.weight *= 5/3

    parameters = [embed_mat] + [p for layer in layers for p in layer.parameters()]
    print("number of parameters:", sum(p.nelement() for p in parameters))
    for p in parameters:
        p.requires_grad = True

    # Training Configuration 
    epochs = 5_000
    learning_rate = 0.07
    decay_drop = 0.95
    decay_epochs = 256
    batch_size = 128
    log_losses = []
    log_update = []

    # Training Iterations
    for e in range(epochs + 1):
        # Create Input
        mini_batch_indices = torch.randint(0, X_train.shape[0], (batch_size, ), generator=g)
        embeds = embed_mat[X_train[mini_batch_indices]]

        # Forward
        x = embeds.view(embeds.shape[0], -1)
        for layer in layers:
            x = layer(x)
        logits = x
        loss = F.cross_entropy(logits, Y_train[mini_batch_indices])
        
        # Backward
        for layer in layers:
            layer.out.retain_grad()
    
        for p in parameters:
            p.grad = None

        loss.backward()

        # Update
        lr = step_decay(e, initial_lrate=learning_rate, drop=decay_drop, epochs_drop=decay_epochs)
        # lr = learning_rate
        for p in parameters:
            p.data -= lr * p.grad

        # Track Stats
        if e % 1_000 == 0:
            print(f"epoch: {e} / {epochs}, lrate: {lr} -> loss: {loss.item()}")
            for idx, layer in enumerate(layers):
                if isinstance(layer, (Linear, Tanh, BatchNorm1d)):
                    dead_neurons = (layer.out.abs() > 0.99).sum()
                    total_neurons = layer.out.view(-1).shape[0]
                    print(f"\tdead neurons ({idx:2d}, {layer.__class__.__name__:20s}): {dead_neurons:10d} / {total_neurons:10d} = {(dead_neurons/total_neurons):.8f}]")

        # Logging Stats
        log_losses.append(loss.log10().item()) 
        with torch.no_grad():
            log_update.append([( lr * p.grad.std() / p.data.std() ).log10().item() for p in parameters if p.grad is not None])


    # Visualize Learning Curve
    plt.figure(figsize=(10, 8))
    plt.title("Learning Curve")
    plt.grid(True)
    plt.plot(np.arange(len(log_losses)), log_losses)
    plt.show()

    # Visualize Update Curve    
    plt.figure(figsize=(20, 5))
    plt.title("Update Curve")
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2:
            plt.plot([log_update[j][i] for j in range(len(log_update))])
            legends.append("parameter %d" % i)
    plt.plot([0, len(log_update)], [-3, -3], "k")
    plt.legend(legends)
    plt.show()
    

    # Visualize Weights
    plt.figure(figsize=(20, 5))
    plt.title("Weights of the Linear Layer")
    legends = []
    for i, p in enumerate(parameters):
        t = p.grad
        if t is None:
            continue
        if p.ndim == 2:
            print("Weight %10s -> mean: %+f, std: %e, grad_data_ratio: %e" % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f"Weight Matrix {i} {tuple(t.shape)}")
    plt.legend(legends)
    plt.show()
    

    # Visualize Histogram 
    # For the Forward Pass
    plt.figure(figsize=(20, 4))
    plt.title("Activation Distribution")
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t = layer.out
            print("Layer %d (%10s): mean %+.4f, std: %.4f, saturated: %.2f%%" 
                % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f"layer {i} ({layer.__class__.__name__})")
    plt.legend(legends)
    plt.show()


    # For the Backward Pass
    plt.figure(figsize=(20, 4))
    plt.title("Gradient Distribution")
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t  = layer.out.grad
            if t is None:
                continue
            print("Layer %d (%10s): mean %+.4f, std: %.4f, saturated: %.2f%%" 
                % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f"layer {i} ({layer.__class__.__name__})")
    plt.legend(legends)
    plt.show()


    # Evaluate Losses on Dataset Splits
    for layer in layers:
        layer.training = False

    split_loss_eval("Training", X_train, Y_train, embed_mat)
    split_loss_eval("Validation", X_val, Y_val, embed_mat)
    split_loss_eval("Test", X_test, Y_test, embed_mat)

    # Sample from the Model 
    for _ in range(20):
        vocab_indices: list[int] = [] 
        context = [0] * block_size
        while True:
            embeds = embed_mat[torch.tensor([context])]
            x = embeds.view(embeds.shape[0], -1)

            for layer in layers:
                x = layer(x)
            logits = x

            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(input=probs, num_samples=1, generator=g).item()
            
            vocab_indices.append(int(idx))
            context = context[1:] + [idx]
            
            if itos[int(idx)] == ".":
                break

        print("".join(itos[vi] for vi in vocab_indices))



