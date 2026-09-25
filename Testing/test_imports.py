import importlib
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

checks = [
    ("torch", None),
    ("numpy", None),
    ("torch.nn", None),
    ("torch.nn.functional", None),
    ("torchvision", "transforms"),
    ("matplotlib.pyplot", None),
    ("utilities3", None),
    ("operator", None),
    ("functools", "reduce"),
    ("functools", "partial"),
    ("timeit", "default_timer"),
    ("scipy.io", None),
    ("os", None),
    ("einops", "rearrange"),
    ("feedforward", "FeedForward"),
    ("linear", "WNLinear"),
    ("xarray", None),
]

failures = []

for module_name, attribute in checks:
    label = f"{module_name}.{attribute}" if attribute else module_name

    try:
        module = importlib.import_module(module_name)
        if attribute:
            getattr(module, attribute)
        print(f"OK:   {label}")
    except Exception as error:
        failures.append(label)
        print(f"FAIL: {label}: {type(error).__name__}: {error}")

if failures:
    print("\nFailed checks:", ", ".join(failures))
    sys.exit(1)

print("\nAll imports succeeded!")

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))