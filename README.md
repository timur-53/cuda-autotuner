# LLM-Powered CUDA Kernel Autotuner

A Python agent that closes the loop between LLM code generation and real GPU
feedback. It asks an LLM to rewrite a CUDA kernel, compiles it with `nvcc`,
verifies the output is correct, benchmarks it with `cudaEvent_t`, and feeds the
results (timings, compile errors, wrong answers) back into the next prompt.

## Results

| Kernel | GPU | Baseline | Best | Speedup |
|--------|-----|----------|------|---------|
| Matmul 1024x1024 (FP32) | Tesla T4 | [BASELINE_MS] ms / [BASELINE_GFLOPS] GFLOPS | 0.687 ms / 3,125 GFLOPS | **[SPEEDUP]x** |

Best of several runs. Baseline and best are measured on the same T4.
The winning kernel is in [`results/best.cu`](results/best.cu) and every attempt
(including rejected ones) is logged in [`results/log.jsonl`](results/log.jsonl).

## How it works

1. Start from a naive kernel (`kernels/naive_matmul.cu`).
2. Ask the LLM for an optimized rewrite.
3. Compile with `nvcc`. Compile errors are sent back to the LLM.
4. Check the output against a double-precision reference kernel. Wrong answers are rejected.
5. Time the kernel with `cudaEvent_t` (average of 20 runs after warmup).
6. Feed timings and errors into the next prompt, then repeat.

`harness.cu` is fixed and never edited by the LLM, so it cannot cheat on timing
or correctness.

## What the LLM discovered

The winning kernel (`results/best.cu`) combines several classic GEMM techniques:

- **Shared-memory tiling:** each block computes a 128x64 output tile, staging
  16-wide slices of A and B in shared memory.
- **Register blocking:** each of the 256 threads computes an 8x4 patch of
  outputs, reusing every loaded value many times.
- **Double buffering:** two shared-memory buffers let the next tile load while the
  current one is being computed.
- **Vectorized `float4` loads** from global memory, used when N is a multiple of 128.
- **A bounds-checked fallback path** for other sizes, selected in `launch()`.

## Run it

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...
    python autotuner.py kernels/naive_matmul.cu --iters 8 --variants 3 --arch sm_75

Requires an NVIDIA GPU and the CUDA toolkit. Developed on Google Colab (Tesla T4).

## Project layout

- `autotuner.py`: the agent loop (prompting, compiling, benchmarking, logging)
- `harness.cu`: fixed correctness check and `cudaEvent_t` timing
- `kernels/naive_matmul.cu`: the baseline kernel
- `results/`: best kernel and run logs

## Limitations and next steps

- Correctness is currently verified at N=1024 only, which exercises the fast
  path. The bounds-checked fallback (N not a multiple of 128) is untested. Next:
  test additional sizes (e.g. N=1000, 1537).
- One kernel type so far (matmul). Next: reduction, softmax, transpose.
- Next: `compute-sanitizer` checks for memory errors and races, a cuBLAS
  comparison, and a Triton backend.
