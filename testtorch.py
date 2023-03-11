import torch
query = torch.rand(5, 10)
search = torch.rand(5, 10)
labels = torch.randint(2, (5, 1))
positive = torch.where(labels == 1, search, query)
for i in range(labels.shape[0]):
    if i == 0:
        if labels[i] == 0:
            negative = search[:-1,:].unsqueeze(0)
        else:
            negative = search[1:,:].unsqueeze(0)
    elif labels[i] == 0:
        negative = torch.cat([negative, search[1:,:].unsqueeze(0)], dim=0)
    elif labels[i] == 1:
        idx = [j for j in range(5) if j != i]
        negative = torch.cat([negative, search[idx,:].unsqueeze(0)], dim=0)
print(query)
print(positive)
print(negative.shape)
print(labels)