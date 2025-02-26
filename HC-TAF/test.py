import random
import torch

a = torch.LongTensor(random.sample(range(0,5), 3))

print (a)

b = torch.tensor([[1.,2,2,2,1],[2,3,4,2,1],[3,2,1,1,1]])
c = b[0].unsqueeze(0)

print(c)