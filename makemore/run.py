#!/bin/python

from pprint import pprint
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    words = open('./names.txt', 'r').read().splitlines()
    print('len:', len(words), 'min:', min(len(w) for w in words), 'max:', max(len(w) for w in words))

    chars = ['.'] + sorted(list(set(''.join(words)))) 
    stoi = {s:i for i,s in enumerate(chars)}
    itos = {v:k for k,v in stoi.items()}

    # create dataset
    xs, ys = [], []

    for w in words[:]:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            idx1, idx2 = stoi[ch1], stoi[ch2]
            xs.append(idx1)
            ys.append(idx2)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    xs_size = xs.nelement() 
    print(f'number of examples: {xs_size}')
    
    # train
    g = torch.Generator().manual_seed(1234)
    W = torch.randn((27,27), generator=g, requires_grad=True)

    epochs = 100
    step_size = 50
    step_size_ten = torch.scalar_tensor(-step_size)

    for e in range(epochs):
        xs_encoded = F.one_hot(xs, num_classes=len(chars)).float()
        logits = xs_encoded @ W

        #probs = F.softmax(logits, dim=1)
        counts = logits.exp() 
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(xs_size), ys].log().mean()

        W.grad = None
        loss.backward()

        W.data += W.grad * step_size_ten

        print(f'{e=}, loss: {loss.data}')

    # generate
    for _ in range(100):
        out = []
        idx = 0
        while True:
            x_encoded = F.one_hot(torch.tensor([idx]), num_classes=(len(chars))).float()
            logits = x_encoded @ W
            ps = F.softmax(logits, dim=1)

            idx = int(torch.multinomial(input=ps, num_samples=1).item())
            out.append(itos[idx])

            if itos[idx] == '.':
                break

        print(''.join(out))


