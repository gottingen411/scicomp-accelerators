# scicomp-accelerators

Exploring classical PDE solvers on modern ML hardware — benchmarking numerical methods for the 1D heat equation across CPU, GPU, and TPU using PyTorch and JAX.

---

## Notebooks

### 1. `heat_equation_pytorch.ipynb` — Crank-Nicolson with PyTorch
Implements the 1D heat equation from first principles using finite difference discretization and the Crank-Nicolson implicit time-stepping scheme with Neumann (zero-flux) boundary conditions.

- Derives and builds the tridiagonal system `A·u^{n+1} = B·u^n`
- Verifies physical correctness: heat conservation to numerical precision
- Tests multiple initial conditions: Gaussian pulse, step function, sinusoidal
- Benchmarks NumPy vs PyTorch CPU vs NVIDIA GPU (Tesla T4)

**Results:**
| Method | Time |
|---|---|
| NumPy | 9317 ms |
| PyTorch CPU | 4601 ms |
| PyTorch GPU | 2006 ms (**4.65x** over NumPy) |

---

### 2. `heat_equation_ftcs_vs_crank_nicolson.ipynb` — Explicit vs Implicit on CPU/GPU
Systematic comparison of two solver strategies on the same problem:

- **Explicit FTCS** (Forward Time Central Space): simple, fast per step, conditionally stable (CFL constraint)
- **Crank-Nicolson**: unconditionally stable, allows larger time steps, requires a linear solve per step

**Results (1000 pts, 2773 steps):**
| Method | CPU | GPU | GPU Speedup |
|---|---|---|---|
| Crank-Nicolson | 56,152 ms | 27,406 ms | 2.1x |
| Explicit FTCS | 1,033 ms | 106 ms | **9.7x** |

Explicit methods parallelize extremely well on GPU — each step is a pure matrix multiply. Implicit methods solve a linear system every step, which is harder to parallelize, explaining the modest GPU gain.

---

### 3. `heat_equation_jax_tpu.ipynb` — JAX with lax.scan on TPU
Re-implements the same solvers in JAX targeting Google TPU. Key focus: `jax.lax.scan` as the correct way to compile iterative loops for accelerators.

- `jax.jit` + `lax.scan` compiles the loop body once into an XLA `while` op — no Python overhead, no graph unrolling
- Compared against uncompiled Python loops to quantify the benefit
- Documents TPU-specific constraints (float32 requirement for LU decomposition)

**Results (CPU, JAX 0.7.2):**
| Variant | Time | Speedup |
|---|---|---|
| Explicit — python loop | 630 ms | baseline |
| Explicit — jit + lax.scan | 295 ms | 2.1x |
| CN — python loop | 83,608 ms | baseline |
| CN — jit + lax.scan | 4,224 ms | **19.8x** |

The 20x speedup on CN highlights how much Python dispatch overhead dominates when each step is expensive — `lax.scan` eliminates it entirely.

---

## Key Takeaways

- Explicit methods GPU-parallelize better than implicit (9.7x vs 2.1x) because matmul scales better than linear solves
- `torch.compile` on iterative solvers can make things *worse* — dynamo unrolls the full loop into a static graph, creating overhead without any benefit when there is a strict sequential data dependency
- `jax.lax.scan` is the right solution: compiles the loop body once, drives it natively in XLA
- TPU requires float32; float64 LU decomposition is not implemented in hardware

## Stack

Python · PyTorch · JAX · NumPy · Matplotlib · Google Colab (T4 GPU / TPU)
