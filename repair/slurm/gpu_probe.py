"""Check VGG11 float32 training memory and CUDA compatibility before downloading data."""

import json
import time

import torch
from repair import vgg11

torch.set_num_threads(4)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
device = torch.device("cuda")
model = vgg11().to(device).train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)
inputs = torch.randn(500, 3, 32, 32, device=device)
labels = torch.randint(10, (500,), device=device)
seconds = []
for step in range(4):
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    seconds.append(time.perf_counter() - started)
    print(f"step={step} loss={loss.item():.6f} seconds={seconds[-1]:.3f}", flush=True)
print(json.dumps({
    "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
    "batch_size": 500, "step_seconds": seconds,
    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    "purpose": "Synthetic input CUDA/memory check only; not CIFAR timing or accuracy",
}), flush=True)
