# scicomp-accelerators

I trained as a chemical engineer, spent years building physics-based simulation models for manufacturing processes, and eventually found myself deep in the world of ML hardware — optimizing LLM inference on Amazon's custom AI chips (Trainium). These feel like separate careers. This repo is my attempt to connect them.

The question I keep coming back to: *what does it look like to run classical numerical simulation on the same hardware that powers modern AI?* GPU and TPU architectures were designed for the dense matrix operations at the heart of deep learning — but those same operations are exactly what PDE solvers need. This repo explores that overlap, using the 1D heat equation as a concrete testbed.

This is exploratory work. The point isn't to produce a production solver — it's to understand the performance landscape and build intuition for where ML accelerators help, where they don't, and why.

---

## Notebooks

### 1. `heat_equation_pytorch.ipynb` — Crank-Nicolson solver with PyTorch

A from-scratch implementation of the heat equation using finite difference discretization and the Crank-Nicolson implicit time-stepping scheme — the same class of methods I used professionally for years, now expressed in PyTorch instead of Fortran.

- Derives and builds the tridiagonal system `A·u^{n+1} = B·u^n` with Neumann (zero-flux) boundary conditions
- Verifies physical correctness: heat conservation to numerical precision across 500 time steps
- Tests multiple initial conditions: Gaussian pulse, step function, sinusoidal
- Benchmarks NumPy vs PyTorch CPU vs NVIDIA GPU (Tesla T4)

**Results:**
| Backend | Time | Speedup |
|---|---|---|
| NumPy | 9,317 ms | baseline |
| PyTorch CPU | 4,601 ms | 2x |
| PyTorch GPU | 2,006 ms | **4.65x** |

The GPU advantage here is modest — the bottleneck is LU factorization, which is harder to parallelize than a pure matmul. A recurring theme in this repo.

---

### 2. `heat_equation_ftcs_vs_crank_nicolson.ipynb` — Explicit vs Implicit across CPU and GPU

Compares two fundamentally different time-stepping strategies on the same physical problem:

- **Explicit FTCS** (Forward Time Central Space): simple, cheap per step, but constrained by the CFL stability condition — requires small time steps
- **Crank-Nicolson**: unconditionally stable, allows larger time steps, but requires solving a linear system at every step

This trade-off is familiar to anyone who has worked with simulation solvers — and the GPU benchmark reveals *why* it matters for hardware choice.

**Results (1000 spatial points, 2773 time steps):**
| Method | CPU | GPU | GPU Speedup |
|---|---|---|---|
| Crank-Nicolson | 56,152 ms | 27,406 ms | 2.1x |
| Explicit FTCS | 1,033 ms | 106 ms | **9.7x** |

Explicit methods are a near-perfect fit for GPU: each step is a pure matrix-vector multiply, massively parallelizable. Implicit methods require a linear solve per step — sequential by nature, and the GPU gives much less leverage. The choice of numerical method isn't just about accuracy and stability; it determines how well your solver maps to the hardware.

---

### 3. `heat_equation_jax_tpu.ipynb` — JAX with lax.scan on TPU

The same solvers reimplemented in JAX, targeting Google TPU. The central question: can JAX's compilation model do what `torch.compile` couldn't?

In an earlier experiment (see `compile_benchmark.py`), `torch.compile` on the full solver loop made things *slower* — because PyTorch's compiler unrolled `range(2773)` into a static graph of 2773 nodes, creating overhead without any parallelism benefit. JAX solves this correctly with `jax.lax.scan`: it compiles the loop body once and drives it via an XLA native `while` loop, eliminating Python overhead without graph explosion.

**Results (CPU, JAX 0.7.2):**
| Variant | Time | Speedup |
|---|---|---|
| Explicit — python loop | 630 ms | baseline |
| Explicit — jit + lax.scan | 295 ms | 2.1x |
| CN — python loop | 83,608 ms | baseline |
| CN — jit + lax.scan | 4,224 ms | **19.8x** |

The 20x speedup on CN is striking. When each time step is expensive (a full linear solve), the Python dispatch overhead accumulated over 2773 steps completely dominates. `lax.scan` eliminates it in one shot. This also hints at why JAX is particularly well-suited for iterative physics solvers — and why TPU, designed around XLA from the ground up, is a natural target for this kind of work.

*Note: TPU requires float32 — LU decomposition is not implemented in float64 on TPU hardware.*

---

## Recurring Themes

**The numerical method determines the hardware story.** Explicit schemes parallelize beautifully on GPU. Implicit schemes don't — and no amount of compiler optimization changes that. The right question isn't "how do I make my solver faster on GPU?" but "does my solver's structure actually map to what the hardware is good at?"

**`torch.compile` doesn't help iterative solvers — `jax.lax.scan` does.** The sequential data dependency `u_{n+1} = f(u_n)` means no iteration can start before the previous one finishes. Compilers can't parallelize what physics won't allow. What they *can* do is eliminate overhead — and `lax.scan` does this correctly by keeping the compiled graph constant-size regardless of iteration count.

**GPU gives 10x on explicit, 2x on implicit.** If you're designing a simulation pipeline that needs to run fast on accelerators, this is the number to keep in mind.

---

## Stack

Python · PyTorch · JAX · NumPy · Matplotlib · Google Colab (T4 GPU / TPU v2)
