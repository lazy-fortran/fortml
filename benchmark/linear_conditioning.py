#!/usr/bin/env python3
"""Run the linear-regression conditioning benchmark against an 80-digit oracle."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import mpmath as mp


EPSILONS = ("1", "1e-4", "1e-8", "1e-12", "1e-14")
TRUE_COEF = ("1.25", "2", "-3", "0.75")
ROW_RE = re.compile(
    r"^linear_conditioning,([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)$"
)
RESIDUAL_RE = re.compile(r"^linear_conditioning_residual,([^,]+)$")


def oracle(epsilon_text: str, dps: int) -> tuple[mp.mpf, list[mp.mpf]]:
    mp.mp.dps = dps
    epsilon = mp.mpf(epsilon_text)
    matrix = mp.matrix(16, 4)
    target = mp.matrix(16, 1)
    coefficients = [mp.mpf(value) for value in TRUE_COEF]
    for row in range(16):
        t = mp.mpf(row + 1) - mp.mpf("8.5")
        alternating = mp.mpf(1 if (row + 1) % 2 == 0 else -1)
        x1 = t
        x2 = t + epsilon * alternating
        x3 = t * t
        matrix[row, 0] = 1
        matrix[row, 1] = x1
        matrix[row, 2] = x2
        matrix[row, 3] = x3
        target[row] = sum(
            coefficient * feature
            for coefficient, feature in zip(coefficients, (1, x1, x2, x3))
        )
    singular_values = mp.svd(matrix, compute_uv=False)
    condition = singular_values[0] / singular_values[-1]
    solution = mp.qr_solve(matrix, target)[0]
    return condition, [solution[index] for index in range(4)]


def parse_rows(output: str) -> tuple[list[dict[str, str]], list[mp.mpf]]:
    rows: list[dict[str, str]] = []
    residuals: list[mp.mpf] = []
    for line in output.splitlines():
        match = ROW_RE.match(line.strip())
        if match:
            values = match.groups()
            rows.append(
                {
                    "epsilon": values[0],
                    "seconds_per_fit": values[1],
                    "coef_0": values[2],
                    "coef_1": values[3],
                    "coef_2": values[4],
                    "coef_3": values[5],
                }
            )
            continue
        match = RESIDUAL_RE.match(line.strip())
        if match:
            residuals.append(mp.mpf(match.group(1)))
    if len(rows) != len(EPSILONS) or len(residuals) != len(EPSILONS):
        raise RuntimeError(f"unexpected benchmark output:\n{output}")
    return rows, residuals


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    compiler = args.compiler or os.environ.get("FORTML_FC") or shutil.which("nvfortran") or "gfortran"
    flags = args.flags or ("-O3 -mp=multicore" if Path(compiler).name == "nvfortran" else "-O3 -march=native")
    build_dir = Path(tempfile.mkdtemp(prefix="fortml-linear-conditioning-"))
    command = [
        "/usr/bin/time",
        "-v",
        "fpm",
        "run",
        "--target",
        "fortml_bench_linear_conditioning",
        "--profile",
        "release",
        "--flag",
        flags,
        "--build-dir",
        str(build_dir),
    ]
    environment = os.environ.copy()
    environment["FPM_FC"] = compiler
    environment["FPM_BUILD_DIR"] = str(build_dir)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    complete_seconds = time.perf_counter() - started
    rows, residuals = parse_rows(completed.stdout)
    rss_match = re.search(
        r"Maximum resident set size \(kbytes\):\s+(\d+)", completed.stderr
    )
    peak_rss_kbytes = int(rss_match.group(1)) if rss_match else -1
    binary_sizes = []
    for build_root in (build_dir,):
        binary_sizes.extend(
            path.stat().st_size
            for path in build_root.rglob("fortml_bench_linear_conditioning")
            if path.is_file()
        )
    generated_code_bytes = max(binary_sizes, default=-1)
    shutil.rmtree(build_dir, ignore_errors=True)
    compiler_version = next(
        line
        for line in subprocess.check_output(
            [compiler, "--version"], text=True, stderr=subprocess.STDOUT
        ).splitlines()
        if line.strip()
    )
    output: list[dict[str, object]] = []
    for index, (row, residual) in enumerate(zip(rows, residuals)):
        condition, expected = oracle(EPSILONS[index], args.dps)
        actual = [mp.mpf(row[f"coef_{column}"]) for column in range(4)]
        coefficient_error = max(
            abs(value - reference) for value, reference in zip(actual, expected)
        )
        coefficient_scale = max(mp.mpf(1), *(abs(value) for value in expected))
        oracle_rows = _oracle_rows(EPSILONS[index], args.dps)
        prediction_error = max(
            abs(sum(value * feature for value, feature in zip(actual, features)) - target)
            for features, target in oracle_rows
        )
        prediction_scale = max(mp.mpf(1), *(abs(target) for _, target in oracle_rows))
        coefficient_relative_error = coefficient_error / coefficient_scale
        prediction_relative_error = prediction_error / prediction_scale
        if prediction_relative_error > mp.mpf("1e-12") or residual > mp.mpf("1e-10"):
            raise RuntimeError(
                f"independent high-precision prediction check failed for epsilon {EPSILONS[index]}"
            )
        if index < 3 and coefficient_relative_error > mp.mpf("1e-5"):
            raise RuntimeError(
                f"independent high-precision coefficient check failed for epsilon {EPSILONS[index]}"
            )
        output.append(
            {
                "compiler": compiler,
                "compiler_version": compiler_version,
                "flags": flags,
                "epsilon": EPSILONS[index],
                "condition_2": mp.nstr(condition, 18),
                "coefficient_relative_error": mp.nstr(coefficient_relative_error, 18),
                "prediction_relative_error": mp.nstr(prediction_relative_error, 18),
                "fortran_residual": mp.nstr(residual, 18),
                "seconds_per_fit": row["seconds_per_fit"],
                "complete_workload_seconds": f"{complete_seconds:.9g}",
                "peak_rss_kbytes": peak_rss_kbytes,
                "generated_code_bytes": generated_code_bytes,
                "mpmath_dps": args.dps,
                "status": "pass",
            }
        )
    return output


def _oracle_rows(epsilon_text: str, dps: int) -> list[tuple[list[mp.mpf], mp.mpf]]:
    mp.mp.dps = dps
    epsilon = mp.mpf(epsilon_text)
    coefficients = [mp.mpf(value) for value in TRUE_COEF]
    rows = []
    for row in range(16):
        t = mp.mpf(row + 1) - mp.mpf("8.5")
        alternating = mp.mpf(1 if (row + 1) % 2 == 0 else -1)
        features = [mp.mpf(1), t, t + epsilon * alternating, t * t]
        target = sum(coefficient * feature for coefficient, feature in zip(coefficients, features))
        rows.append((features, target))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler")
    parser.add_argument("--flags")
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dps < 50:
        raise SystemExit("--dps must be at least 50")
    rows = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
