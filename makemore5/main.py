#!/usr/bin/env python

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from makemore_lib import *

if __name__ == "__main__":
    # Seeding Generators
    random.seed(42)
    g = torch.manual_seed(42)

    # Dataset Preparation
    words = open("./names.txt").read().splitlines()
    chars = ["."] + sorted(list(set("".join(words))))
    stoi = {char: idx for idx, char in enumerate(chars)}
    itos = {idx: char for char, idx in stoi.items()}

    training_percentage = 0.8
    validation_percentage = 0.1
    test_percentage = 0.1

    n1 = int(training_percentage * len(words))
    n2 = int((training_percentage + validation_percentage) * len(words))

    # Context Window
    block_size = 8

    random.shuffle(words) 
    X_train, Y_train = build_dataset(words[:n1], block_size, stoi)    # 80%
    X_val, Y_val = build_dataset(words[n1:n2], block_size, stoi)      # 10%
    X_test, Y_test = build_dataset(words[n2:], block_size, stoi)      # 10%

    # Layers
    vocab_size = len(chars)
    n_embedding = 16
    n_hidden = 128
    
    model = Sequential([
        Embedding(vocab_size, n_embedding),

        FlattenConsecutive(2),
        Linear(n_embedding * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),

        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),

        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),

        Linear(n_hidden, vocab_size, bias=False),
    ])

    # General Layers Setup
    with torch.no_grad():
        if isinstance(model.layers[-1], Linear):
            model.layers[-1].weight *= 0.1


    parameters = model.parameters()
    print("number of parameters:", sum(p.nelement() for p in parameters))
    for p in parameters:
        p.requires_grad = True


    # Training Configuration 
    epochs = 10 * 1000 
    batch_size = 128
    log_losses = []
    log_update = []

    # Training Iterations
    for e in range(epochs):
        # Create Input
        mini_batch_indices = torch.randint(0, X_train.shape[0], (batch_size, ) )

        # Forward
        logits = model(X_train[mini_batch_indices])
        loss = F.cross_entropy(logits, Y_train[mini_batch_indices])
        
        # Backward
        for layer in model.layers:
            layer.out.retain_grad()

        for p in parameters:
            p.grad = None

        loss.backward()

        # Update
        lr = step_decay(e, initial_lrate=0.1, drop=0.5, epochs_drop=5_000)
        for p in parameters:
            p.data -= torch.scalar_tensor(lr) * p.grad

        # Logging Stats
        log_losses.append(loss.item()) 
        with torch.no_grad():
            log_update.append([( lr * p.grad.std() / p.data.std() ).log10().item() for p in parameters if p.grad is not None])

        # Track Stats
        track_step = 1_000
        if e % track_step == 0:
            start = 0 if e == 0 else e - track_step
            mean_loss = torch.tensor(log_losses[start:e + 1]).mean()
            print(f"epoch: {e} / {epochs}, lrate: {lr}, [{start}:{e}] -> loss: {mean_loss}")

            for idx, layer in enumerate(model.layers):
                if isinstance(layer, Layer):
                    dead_neurons = (layer.out.abs() > 0.99).sum()
                    total_neurons = layer.out.view(-1).shape[0]
                    print(f"\tdead neurons ({idx:2d}, {layer.__class__.__name__:20s}): {dead_neurons:10d} / {total_neurons:10d} = {(dead_neurons/total_neurons):.8f}]")


    # Visualize Learning Curve
    plt.figure(figsize=(10, 8))
    plt.title("Learning Curve")
    plt.grid(True)
    plt.plot(torch.tensor(log_losses).log10().view(-1, 100).mean(dim=1))
    plt.show()

    # Visualize Update Curve    
    plt.figure(figsize=(20, 5))
    plt.title("Update Curve")
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2:
            plt.plot(torch.tensor([log_update[j][i] for j in range(len(log_update))]).view(-1, 100).mean(dim=1))
            legends.append("parameter %d" % i)
    plt.plot([0, len(log_update)/100], [-3, -3], "k")
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
    for i, layer in enumerate(model.layers[:-1]):
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
    for i, layer in enumerate(model.layers[:-1]):
        if isinstance(layer, Linear):
            t = layer.out.grad
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
    for layer in model.layers:
        layer.training = False

    split_loss_eval("Training", X_train, Y_train, model)
    split_loss_eval("Validation", X_val, Y_val, model)
    split_loss_eval("Test", X_test, Y_test, model)

    # Sample from the Model 
    for _ in range(20):
        vocab_indices: list[int] = [] 
        context = [0] * block_size

        while True:
            logits = model(torch.tensor(context).view(1,-1))
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(input=probs, num_samples=1).item()
            
            vocab_indices.append(int(idx))
            context = context[1:] + [idx]
            
            if itos[int(idx)] == ".":
                break

        print("".join(itos[vi] for vi in vocab_indices))



