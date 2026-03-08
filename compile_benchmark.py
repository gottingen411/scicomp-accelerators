"""
Benchmark torch.compile strategies for the explicit heat equation solver.

Approaches:
  1. Uncompiled (baseline)
  2. Compile step only  (what the notebook did)
  3. Compile full loop  (the fix)
  4. Compile full loop, max-autotune mode
"""
import time
import torch

torch.set_default_dtype(torch.float64)

# ── Parameters (same as notebook) ────────────────────────────────────────────
L, alpha, T_total = 1.0, 0.00025, 5.0
nx      = 1000
dx      = L / (nx - 1)
r       = 0.45 * dx**2 / alpha     # dt from CFL
dt      = r * dx**2 / alpha
nt      = int(T_total / dt) + 1
r_exp   = alpha * dt / dx**2

print(f"Grid: {nx} pts, {nt} steps, r={r_exp:.4f}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
print()

# ── Build matrix D ────────────────────────────────────────────────────────────
def build_D(nx, r, device):
    D = torch.zeros(nx, nx, device=device)
    for i in range(1, nx - 1):
        D[i, i-1], D[i, i], D[i, i+1] = r, 1 - 2*r, r
    D[0,  0],  D[0,  1]  = 1 - 2*r, 2*r
    D[-1, -2], D[-1, -1] = 2*r,     1 - 2*r
    return D

# ── Solver variants ───────────────────────────────────────────────────────────
def solve_uncompiled(D, u0, nt):
    u = u0.clone()
    for _ in range(nt):
        u = D @ u
    return u

def _step(D, u):          # compiled step only
    return D @ u

def solve_step_compiled(D, u0, nt, step_fn):
    u = u0.clone()
    for _ in range(nt):
        u = step_fn(D, u)
    return u

def solve_full_loop(D, u0, nt):   # entire loop compiled
    u = u0.clone()
    for _ in range(nt):
        u = D @ u
    return u

# ── Benchmark helper ──────────────────────────────────────────────────────────
def bench(fn, *args, n_runs=3, warmup=1, cuda=False):
    for _ in range(warmup):
        out = fn(*args)
        if cuda: torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        if cuda: torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return out, min(times) * 1000   # best-of-n in ms

# ── Run ───────────────────────────────────────────────────────────────────────
compiled_step      = torch.compile(_step)
compiled_full      = torch.compile(solve_full_loop)
compiled_full_fast = torch.compile(solve_full_loop, mode="max-autotune")

header = f"{'Method':<35} {'Device':<6} {'ms':>10}"
sep    = "-" * len(header)

print(header)
print(sep)

for device_str in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
    device = torch.device(device_str)
    cuda   = device_str == "cuda"
    D      = build_D(nx, r_exp, device)
    x      = torch.linspace(0, L, nx, device=device)
    u0     = torch.exp(-((x - 0.5)**2) / (2*(0.1)**2))

    label  = "GPU" if cuda else "CPU"

    _, t = bench(solve_uncompiled, D, u0, nt, cuda=cuda)
    print(f"{'Uncompiled':<35} {label:<6} {t:>10.1f}")

    _, t = bench(solve_step_compiled, D, u0, nt, compiled_step, cuda=cuda)
    print(f"{'compile(step only)':<35} {label:<6} {t:>10.1f}")

    _, t = bench(compiled_full, D, u0, nt, cuda=cuda)
    print(f"{'compile(full loop)':<35} {label:<6} {t:>10.1f}")

    _, t = bench(compiled_full_fast, D, u0, nt, cuda=cuda)
    print(f"{'compile(full loop, max-autotune)':<35} {label:<6} {t:>10.1f}")

    print(sep)
