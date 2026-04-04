"""GPU utilization and throughput benchmarks."""

import time
import torch

from hexo.game.constants import BOARD_SIZE
from hexo.model.resnet import HexONet


def benchmark_forward(model=None, device=None, batch_sizes=(1, 8, 32, 64, 128),
                      n_iters=100, warmup=10):
    """Benchmark model forward pass throughput."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = HexONet().to(device)
    model.eval()

    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, BOARD_SIZE, BOARD_SIZE, device=device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_per_call = elapsed / n_iters * 1000
        samples_per_sec = bs * n_iters / elapsed
        results[bs] = {
            'ms_per_call': ms_per_call,
            'samples_per_sec': samples_per_sec,
        }
        print(f"  B={bs:3d}: {ms_per_call:.2f}ms/call  "
              f"{samples_per_sec:.0f} samples/s")

    return results


def benchmark_forward_fp16(model=None, device=None,
                           batch_sizes=(1, 32, 128), n_iters=200):
    """Benchmark with AMP autocast."""
    if device is None:
        device = torch.device('cuda')
    if model is None:
        model = HexONet().to(device)
    model.eval()

    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, BOARD_SIZE, BOARD_SIZE, device=device)

        for _ in range(10):
            with torch.no_grad(), torch.amp.autocast('cuda'):
                model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad(), torch.amp.autocast('cuda'):
                model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_per_call = elapsed / n_iters * 1000
        samples_per_sec = bs * n_iters / elapsed
        results[bs] = {
            'ms_per_call': ms_per_call,
            'samples_per_sec': samples_per_sec,
        }
        print(f"  FP16 B={bs:3d}: {ms_per_call:.2f}ms/call  "
              f"{samples_per_sec:.0f} samples/s")

    return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HexONet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params on {device}")

    print("\nFP32 throughput:")
    benchmark_forward(model, device)

    if device.type == 'cuda':
        print("\nFP16 throughput:")
        benchmark_forward_fp16(model, device)
