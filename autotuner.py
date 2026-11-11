#!/usr/bin/env python3
"""LLM-powered CUDA kernel autotuner: generate -> compile -> verify -> benchmark -> feedback.

Usage:
    export ANTHROPIC_API_KEY=...
    python autotuner.py kernels/naive_matmul.cu --iters 8 --variants 3
"""
import argparse, json, re, subprocess, time, shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent
MODEL = "claude-sonnet-5-5"  # swap for any model / local endpoint

SYSTEM = """You are an expert CUDA performance engineer. You rewrite a CUDA kernel file
to make it faster while keeping results correct. Hard rules:
- Keep the exact interface: `void launch(const float* A, const float* B, float* C, int N);`
- Row-major NxN float matmul, N not guaranteed to be a multiple of tile size (handle bounds).
- No external libraries (no cuBLAS). Single .cu file, include <cuda_runtime.h>.
- Return the FULL file in one ```cuda code block, preceded by 1-3 sentences on what you changed and why.
Optimization ideas: shared-memory tiling, block/tile size changes, register blocking
(each thread computes multiple outputs), loop unrolling (#pragma unroll), memory coalescing,
avoiding bank conflicts (padding), vectorized loads (float4)."""


@dataclass
class Result:
    stage: str            # "compile" | "run" | "incorrect" | "ok"
    time_ms: float | None = None
    gflops: float | None = None
    detail: str = ""


def compile_kernel(src: Path, out: Path, arch: str) -> tuple[bool, str]:
    cmd = ["nvcc", "-O3", f"-arch={arch}", str(ROOT / "harness.cu"), str(src), "-o", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode == 0, p.stderr[-2000:]


def benchmark(binary: Path, n: int, timeout: int = 60) -> Result:
    try:
        p = subprocess.run([str(binary), str(n)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result("run", detail="timeout")
    try:
        j = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return Result("run", detail=f"bad output: {p.stdout[-300:]} {p.stderr[-300:]}")
    if not j["ok"]:
        return Result("incorrect" if "incorrect" in j["error"] else "run", detail=json.dumps(j))
    return Result("ok", j["time_ms"], j["gflops"])


def evaluate(src: Path, tag: str, n: int, arch: str) -> Result:
    binary = src.with_suffix(".bin")
    ok, err = compile_kernel(src, binary, arch)
    if not ok:
        return Result("compile", detail=err)
    return benchmark(binary, n)


def extract_code(text: str) -> str | None:
    m = re.search(r"```(?:cuda|cpp|c\+\+)?\n(.*?)```", text, re.S)
    return m.group(1) if m else None


def build_prompt(history: list[dict], best: dict) -> str:
    lines = [f"Current BEST kernel ({best['res']['time_ms']} ms, {best['res']['gflops']} GFLOPS):",
             f"```cuda\n{best['code']}\n```", "", "Recent attempts and outcomes:"]
    for h in history[-6:]:
        r = h["res"]
        summary = (f"{r['time_ms']} ms / {r['gflops']} GFLOPS" if r["stage"] == "ok"
                   else f"{r['stage'].upper()} FAILURE: {r['detail'][:400]}")
        lines.append(f"- {h['desc']} -> {summary}")
    lines.append("\nPropose ONE improved version of the best kernel. Try something not already attempted.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--variants", type=int, default=3, help="candidates per iteration")
    ap.add_argument("-n", type=int, default=1024)
    ap.add_argument("--arch", default="native", help="nvcc -arch value, e.g. sm_80 or native")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    log = (run_dir / "log.jsonl").open("a")
    client = anthropic.Anthropic()

    base_src = run_dir / "v000_baseline.cu"
    shutil.copy(args.kernel, base_src)
    res = evaluate(base_src, "baseline", args.n, args.arch)
    print(f"[baseline] {res}")
    if res.stage != "ok":
        raise SystemExit("Baseline must compile and pass correctness.")
    best = {"code": base_src.read_text(), "res": asdict(res)}
    baseline_ms = res.time_ms
    history: list[dict] = []
    vid = 0

    for it in range(1, args.iters + 1):
        for _ in range(args.variants):
            vid += 1
            msg = client.messages.create(
                model=MODEL, max_tokens=4000, temperature=1.0, system=SYSTEM,
                messages=[{"role": "user", "content": build_prompt(history, best)}],
            )
            text = msg.content[0].text
            code = extract_code(text)
            if not code:
                continue
            desc = text.split("```")[0].strip().replace("\n", " ")[:200]
            src = run_dir / f"v{vid:03d}.cu"
            src.write_text(code)
            r = evaluate(src, f"v{vid}", args.n, args.arch)
            rec = {"id": vid, "iter": it, "desc": desc, "res": asdict(r)}
            history.append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            tag = f"{r.time_ms} ms ({baseline_ms / r.time_ms:.1f}x)" if r.stage == "ok" else r.stage
            print(f"[iter {it}] v{vid:03d}: {tag} | {desc[:80]}")
            if r.stage == "ok" and r.time_ms < best["res"]["time_ms"]:
                best = {"code": code, "res": asdict(r)}

    (run_dir / "best.cu").write_text(best["code"])
    print(f"\nBest: {best['res']['time_ms']} ms, {best['res']['gflops']} GFLOPS "
          f"({baseline_ms / best['res']['time_ms']:.1f}x over baseline) -> {run_dir/'best.cu'}")


if __name__ == "__main__":
    main()
