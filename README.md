# LLM-Powered CUDA Kernel Autotuner

A Python agent that closes the loop between LLM code generation and real GPU
feedback. It asks an LLM to rewrite a CUDA kernel, compiles it with `nvcc`,
verifies the output is correct, benchmarks it with `cudaEvent_t`, and feeds the
results (timings, compile errors, wrong answers) back into the next prompt.

## Results

| Kernel | GPU | Baseline | Best | Speedup |
|--------|-----|----------|------|---------|
| Matmul 1024x1024 (FP32) | Tesla T4 | 5.84 ms / 367 GFLOPS | 0.748 ms / 2,870 GFLOPS | **7.8x** |

Setup for this run: 2 iterations x 2 candidates per iteration.
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

- Shared-memory tiling combined with register blocking (each thread computes
  multiple outputs).
- Vectorized `float4` global-to-shared loads.
- A larger 64x128 output tile with 128 threads per block.

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

- Correctness is currently verified at N=1024 only. Next: test additional sizes
  (e.g. N=1000, 1537) to catch tile-boundary bugs.
- One kernel type so far (matmul). Next: reduction, softmax, transpose.
- Next: `compute-sanitizer` checks for memory errors and races, a cuBLAS
  comparison, and a Triton backend.
