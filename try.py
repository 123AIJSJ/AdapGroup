import torch


a = torch.randn(3, 2)
b = torch.randint_like(a, 5)
c = torch.randint_like(a, 5)

print(b)
print(c)
print(torch.norm(b-c, dim=1))


