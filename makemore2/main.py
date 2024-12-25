#!/usr/bin/env python

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

words = open("./names.txt").read().splitlines()
chars = ["."] + sorted(list(set("".join(words))))
stoi = {char: idx for idx, char in enumerate(chars)}
itos = {idx: char for char, idx in stoi.items()}

def build_dataset(words, block_size):
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

def step_decay(epoch, initial_lrate=0.1, drop=0.5, epochs_drop=100):
   return initial_lrate * np.pow(drop, np.floor((1+epoch)/epochs_drop))

if __name__ == "__main__":
    random.seed(42)
    g = torch.Generator().manual_seed(2147483647)

    train_n = 0.8
    dev_n = 0.1
    test_n = 0.1

    n1 = int(train_n * len(words))
    n2 = int((train_n + dev_n) * len(words))

    block_size = 5
    random.shuffle(words) 
    X_train, Y_train = build_dataset(words[:n1], block_size)    # 80%
    X_dev, Y_dev = build_dataset(words[n1:n2], block_size)      # 10%
    X_test, Y_test = build_dataset(words[n2:], block_size)      # 10%

    # Layers
    vocab_size = len(chars)
    n_embs = 10
    n_hidden = 200
    layer_sz = [
        block_size * n_embs, 
        n_hidden, 
        vocab_size,
    ]

    # Initialize Parameters
    C = torch.randn((vocab_size, n_embs),        generator=g)
    W1 = torch.randn((layer_sz[0], layer_sz[1]), generator=g) * (5/3)/(layer_sz[0]**0.5)
    b1 = torch.randn(layer_sz[1],                generator=g) * 0.01
    W2 = torch.randn((layer_sz[1], layer_sz[2]), generator=g) * 0.01
    b2 = torch.randn(layer_sz[2],                generator=g) * 0

    # Batch Normalization Parameters
    bn_gain = torch.ones((1, n_hidden))
    bn_bias = torch.zeros((1, n_hidden))
    bn_mean_running = torch.zeros((1, n_hidden))
    bn_std_running = torch.ones((1, n_hidden))

    parameters = [
        C, 
        W1, 
        # b1, 
        W2, 
        b2, 
        bn_gain, 
        bn_bias,
    ]

    for p in parameters:
        p.requires_grad = True

    print(f"number of parameters: {sum(p.nelement() for p in parameters)}")

    # Training
    epochs = 10000
    learning_rate = 1
    batch_size = 32 
    
    loss_i = []

    for e in range(epochs):
        # Minibatch
        batch_indices = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
        
        # Forward Pass
        emb = C[X_train[batch_indices]].view(batch_size, -1)

        # Linear Layer 1
        h1 = emb @ W1# + b1

        # Batch Norm Layer 1
        bn_meani = h1.mean(0, keepdim=True)
        bn_stdi = h1.std(0, keepdim=True)
        h1 = bn_gain * (h1 - bn_meani) / bn_stdi + bn_bias

        with torch.no_grad():
            bn_mean_running = 0.999 * bn_mean_running + 0.001 * bn_meani
            bn_std_running = 0.999 * bn_std_running + 0.001 * bn_stdi

        # Non-Linearity Layer 1
        h1_act = F.tanh(h1)

        # Linear Layer 2
        h2 = h1_act @ W2 + b2

        # Loss
        loss = F.cross_entropy(h2, Y_train[batch_indices])

        # Backward Pass
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # Update
        lr = step_decay(e, initial_lrate=learning_rate, drop=0.5, epochs_drop=1000)
        with torch.no_grad():
            for p in parameters:
                p -= lr * p.grad

        # Debug Shit
        # print(f"e: {e} -> loss: {loss.item()}")
        loss_i.append(loss.log10().item())

        if e % 1000 == 0:
            print(f"[e: {e:7d} / {epochs}][lr: {lr:8f}] -> [loss: {loss.item():8f}]")
            dead_neurons = (h1.abs() > 0.99).sum()
            total_neurons = h1.view(-1).shape[0]
            print(f"[dead neurons (layer 1): {dead_neurons} / {total_neurons} = {dead_neurons/total_neurons}]")
            # plt.hist(h1.view(-1).tolist(), 50)
            # plt.show()
            # plt.figure(figsize=(20,10))
            # plt.imshow(h1.abs() > 0.99, cmap="gray", interpolation="nearest")
            # plt.show()

    
    plt.plot(loss_i)
    plt.show()

    # Evaulate the Loss
    emb = C[X_dev].view(X_dev.shape[0], layer_sz[0])
    h1 = emb @ W1 # + b1)

    h1 = bn_gain * (h1 - bn_mean_running) / bn_std_running + bn_bias

    h1_act = F.tanh(h1)

    h2 = h1_act @ W2 + b2
    loss = F.cross_entropy(h2, Y_dev)
    print(">> loss:", loss.item())

    # Text Generation
    for _ in range(20):
        out = []
        context = [0] * block_size

        while True:
            emb = C[torch.tensor([context])].view(1, -1)
            h1 = emb @ W1 # + b1)

            h1 = bn_gain * (h1 - bn_mean_running) / bn_std_running + bn_bias

            h1_act = F.tanh(h1)
            h2 = h1_act @ W2 + b2
            probs = F.softmax(h2, dim=1)

            idx = int(torch.multinomial(probs, num_samples=1, generator=g).item())
            context = context[1:] + [idx]
            out.append(itos[idx])

            if itos[idx] == '.':
                break

        print(''.join(out))

    # plt.figure(figsize=(8,8))
    # plt.scatter(x=C[:,0].data, y=C[:,1].data, s=200)
    # for i in range(C.shape[0]):
    #     plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
    # plt.grid(True)
    # plt.show()
