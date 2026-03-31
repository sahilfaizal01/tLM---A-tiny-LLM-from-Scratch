import torch
from softmax import compiled_softmax, softmax_pytorch

def benchmark(func, inp_tensor, iters=1000, warmup=100):
  # warmup - to trace the graph, compile the kernels
  for _ in range(warmup):
    func(inp_tensor)
  torch.cuda.synchronize() # forces Python to wait for GPU execution to finish
  # measure execution time only
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)

  start.record()
  for _ in range(iters):
    func(inp_tensor)
  end.record()

  torch.cuda.synchronize()
  return start.elapsed_time(end) / iters 


if __name__ == "__main__":
  input_tensor = torch.randn((1024,1024), device="cuda")
  # Benchmarking speed
  default_time = benchmark(softmax_pytorch, input_tensor)
  compiled_time = benchmark(compiled_softmax, input_tensor)
  # Outputs to verify correctness
  default_output = softmax_pytorch(input_tensor)
  compiled_output = compiled_softmax(input_tensor)
  # Print the output
  print("Default Softmax Time (ms):", default_time)
  print("Compiled Softmax Time (ms):", compiled_time)
  print("Speedup:", default_time / compiled_time)
  print("Max difference:", (default_output - compiled_output).abs().max().item
