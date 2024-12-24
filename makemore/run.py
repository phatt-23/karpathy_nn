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

    # occurence = torch.zeros((27,27), dtype=torch.int32)
    # for w in words[:]:
    #     chs = ['.'] + list(w) + ['.']
    #     for c1, c2 in zip(chs, chs[1:]):
    #         i, j = stoi[c1], stoi[c2]
    #         occurence[i,j] += 1

    # plt.figure(figsize=(16,16)) # on i3, pop out the window to floating
    # plt.imshow(occurence, cmap='Greens')
    # for i in range(len(chars)):
    #     for j in range(len(chars)):
    #         c1, c2 = itos[i], itos[j]
    #         plt.text(j, i, c1 + c2,                     ha='center', va='top',    color='gray')
    #         plt.text(j, i, str(occurence[i,j].item()),  ha='center', va='bottom', color='gray')
    # plt.axis('off') 
    # plt.show()
    
    # norm_occ = (occurence).float()
    # norm_occ /= norm_occ.sum(1, keepdim=True)
    #
    # for _ in range(10):
    #     out = []
    #     idx = 0
    #     while True:
    #         p = norm_occ[idx]
    #         idx = int(torch.multinomial(input=p, num_samples=1).item())
    #         out.append(itos[idx])
    #         if itos[idx] == '.':
    #             break
    #     print(''.join(out))
    
    # log_likelihood = 0.0
    # n = 0
    # for w in ['andrejq']:
    #     chs = ['.'] + list(w) + ['.']
    #     for c1, c2 in zip(chs, chs[1:]):
    #         i, j = stoi[c1], stoi[c2]
    #         prob = norm_occ[i,j]
    #         log_prob = torch.log(prob)
    #         log_likelihood += log_prob
    #         n += 1
    #         print(f'{c1}{c2}, {prob=}, {log_prob=}') 
    # 
    # print(f'{log_likelihood=}')
    # nll = -log_likelihood
    # print(f'{nll=}, {nll/n=}')

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


