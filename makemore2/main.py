#!/usr/bin/env python

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    words = open("./names.txt").read().splitlines()
    chars = ["."] + sorted(list(set("".join(words))))
    stoi = {char: idx for idx, char in enumerate(chars)}
    itos = {idx: char for char, idx in stoi.items()}
    
    block_size = 3
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

    print("input:", X.shape, X.dtype)
    print("output:", Y.shape, Y.dtype)

    # Create Embedding
    emb_dim = 2
    C = torch.randn((len(chars), emb_dim), requires_grad=True)

    # Layers
    layer_sz = [
        block_size * emb_dim, 
        100, 
        len(chars)
    ]

    # Initialize W and b
    W1 = torch.randn((layer_sz[0], layer_sz[1]), requires_grad=True)
    b1 = torch.randn(layer_sz[1], requires_grad=True)
    W2 = torch.randn((layer_sz[1], layer_sz[2]), requires_grad=True)
    b2 = torch.randn(layer_sz[2], requires_grad=True)
    parameters = [C, W1, b1, W2, b2]

    # Training
    epochs = 1000
    step_size = torch.tensor(0.1)
    step = torch.tensor(0.001)
    batch_size = 32

    for e in range(epochs):
        batch_indices = torch.randint(0, X.shape[0], (batch_size,))
        
        emb = C[X[batch_indices]].view(batch_size, -1)
        h1 = torch.tanh(emb @ W1 + b1)
        h2 = h1 @ W2 + b2
        loss = F.cross_entropy(h2, Y[batch_indices])

        print(f"e: {e} -> loss: {loss.item()}")

        for p in parameters:
            p.grad = None
        loss.backward()

        with torch.no_grad():
            if e < 100:
                for p in parameters:
                    p -= step_size * p.grad
            else:
                for p in parameters:
                    p -= step * p.grad

    # Text Generation
    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor(context).unsqueeze(0)].view(1, -1)
            h1 = torch.tanh(emb @ W1 + b1)
            h2 = h1 @ W2 + b2
            probs = F.softmax(h2, dim=1)
            idx = int(torch.multinomial(probs, num_samples=1).item())
            out.append(itos[idx])
            context = context[1:] + [idx]
            if itos[idx] == '.':
                break

        print(''.join(out))
