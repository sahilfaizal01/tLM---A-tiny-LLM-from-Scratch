import torch

def softmax_pytorch(x):
  x_max = torch.max(x, dim=-1, keepdim=True).values
  x_exp = torch.exp(x - x_max)
  return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

@torch.compile
def compiled_softmax(x):
  return softmax_pytorch(x)

if __name__ == "__main__":
  input_tensor = torch.randn((2,4), device="cuda")
  output = compiled_softmax(input_tensor)
  print("Input:", input_tensor)
  print("Compiled Softmax Output:", output) 
  
