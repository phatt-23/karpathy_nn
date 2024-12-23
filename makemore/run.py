#!/bin/python

from pprint import pprint
import matplotlib.pyplot as plt
import torch

def main():
    words = open('./names.txt', 'r').read().splitlines()

    # pprint(words[:10])
    print('len:', len(words), 'min:', min(len(w) for w in words), 'max:', max(len(w) for w in words))

    occurence = torch.zeros((27,27), dtype=torch.int32)

    chars = ['.'] + sorted(list(set(''.join(words)))) 
    stoi = {s:i for i,s in enumerate(chars)}
    itos = {v:k for k,v in stoi.items()}

    # pprint(stoi)

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
    
    for 

if __name__ == "__main__":
    main()

