import torch
import numpy as np

# Create a tensor from a list
a = [1,2,3]
t_a = torch.tensor(a)
print(t_a)
print(t_a.shape, t_a.dtype)
print('----------------')

# Create a tensor from a NumPy array
b = np.array([4,5,6], dtype=np.int32)
t_b = torch.tensor(b)
print(t_b)
print(t_b.shape)
print('----------------')

# Create a tensor from a NumPy array with a specific data type
t_a_new = t_a.to(torch.int32)
print(t_a_new)
print(t_a_new.shape, t_a_new.dtype)
print('----------------')

# Reshape a tensor
# 1d tensor to 2d tensor
t_c = torch.zeros(30)
t_reshape = t_c.reshape(5,6)
print(t_c)
print(t_reshape)
print(t_c.shape, t_reshape.shape)
print('----------------')

# Squeeze a tensor: remove unnecessary dimensions
# imensions of size 1, which are unnecessary
t_d = torch.zeros(1,2,1,4,1)
t_d_sqz = torch.squeeze(t_d)
print(t_d.shape, t_d_sqz.shape)
print('----------------')

# Unsqueeze a tensor: add a dimension
t_e = torch.zeros(3,4)
t_e_unsqz = torch.unsqueeze(t_e, 0)
print(t_e.shape, t_e_unsqz.shape)
print('----------------')

# Chunk a tensor: split a tensor into smaller chunks
t_f = torch.rand(10)
t_chunk = torch.chunk(t_f, 3)
print(t_f)
print(t_chunk)
print('----------------')

t_f = torch.rand(10)
t_split = torch.split(t_f, [5,4,1])
print([item.numpy() for item in t_split])
print('----------------')

# Stack tensors
t_cat = torch.cat([item for item in t_split], axis=0)
print(t_cat)
print('----------------')

t_c1 = torch.zeros(3)
t_c2 = torch.ones(3)
t_stack = torch.stack([t_c1, t_c2], axis=1)
print(t_stack)
print('----------------')








