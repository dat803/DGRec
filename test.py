import torch

t1 = torch.tensor([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
])


print(torch.einsum('ab,cd->b', t1, t1))