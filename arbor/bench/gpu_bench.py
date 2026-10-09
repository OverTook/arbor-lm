import subprocess

import torch


def timed(fn, iters):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


def matmul_tflops(m, n, k, dtype=torch.bfloat16):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    t = timed(lambda: a @ b, 50)
    return 2 * m * n * k / t / 1e12


def pcie_gbps(mb=256):
    host = torch.empty(mb << 20, dtype=torch.uint8).pin_memory()
    dev = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
    h2d = timed(lambda: dev.copy_(host, non_blocking=True), 20)
    d2h = timed(lambda: host.copy_(dev, non_blocking=True), 20)
    return (mb << 20) / h2d / 1e9, (mb << 20) / d2h / 1e9


def gpu_state():
    q = "temperature.gpu,power.draw,clocks.sm,pcie.link.gen.current,pcie.link.width.current"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def main():
    print("device", torch.cuda.get_device_name(0), "torch", torch.__version__)
    for shape in [(4096, 4096, 4096), (16384, 2048, 768), (16384, 768, 2048), (16384, 512, 512)]:
        print(f"bf16_matmul {shape} TFLOPS {matmul_tflops(*shape):.1f}")
    print(f"fp32_matmul (4096,)*3 TFLOPS {matmul_tflops(4096, 4096, 4096, torch.float32):.1f}")
    h2d, d2h = pcie_gbps()
    print(f"pcie_h2d_GBps {h2d:.1f} pcie_d2h_GBps {d2h:.1f}")
    print("gpu_state(temp,W,MHz,gen,width)", gpu_state())


if __name__ == "__main__":
    main()
