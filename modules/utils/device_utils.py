import torch


def get_device() -> str:
    """Return the best available device: cuda > xpu > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_available_devices() -> list:
    """Return list of all available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.xpu.is_available():
        devices.append("xpu")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices
