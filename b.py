import torch
import dgl
import pickle
import torchtext

import onmt.myutils as myutils
# dataset = torch.load('data/seqdata.train.0.pt')
# g = myutils.str2graph(dataset.examples[0].src)
# setattr(dataset.examples[0], 'graph', g)
# print(g)
# torch.save(dataset,'g.pt')
# a = torch.load('g.pt')
# print(a.examples[0].graph)
# print(a)
dataset = torch.load('data/seqdata.train.0.pt')
g = myutils.str2graph(dataset.examples[0].src)
setattr(dataset.examples[0], 'graph', g)
with open('g.pt', 'wb') as f:
    pickle.dump(dataset, f)
with open('g.pt', 'rb') as f:
    a = pickle.load(f)

print(a)