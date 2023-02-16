import torch
sequence_output = torch.rand(4, 500, 700)
dim = torch.mean(sequence_output, dim=2).shape
print(dim)