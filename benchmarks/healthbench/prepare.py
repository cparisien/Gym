# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare HealthBench for direct NeMo Gym evaluation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.healthbench_rubric.prepare_common import iter_local_records, load_jsonl, write_gym_jsonl


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "healthbench_benchmark.jsonl"

HF_DATASET = "openai/healthbench"
BENCHMARK_NAME = "healthbench"

SOURCE_FILENAMES = {
    "oss_eval": "2025-05-07-06-14-12_oss_eval.jsonl",
    "oss_meta_eval": "2025-05-07-06-14-12_oss_meta_eval.jsonl",
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",
    "hard": "hard_2025-05-08-21-00-10.jsonl",
}


def _iter_source_records(data_dir: str | Path, subset: str) -> Iterator[dict[str, Any]]:
    source_filename = SOURCE_FILENAMES[subset]
    local_records = iter_local_records(data_dir, source_filename)
    if local_records is not None:
        yield from local_records
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            f"{data_dir!r} is not a local path. Install huggingface_hub or pass a local HealthBench JSONL path."
        ) from error

    downloaded = hf_hub_download(
        repo_id=str(data_dir),
        filename=source_filename,
        repo_type="dataset",
    )
    yield from load_jsonl(Path(downloaded))


def prepare(
    data_dir: str | Path = HF_DATASET,
    output: Path = OUTPUT_FPATH,
    limit: int | None = None,
    subset: str = "oss_eval",
) -> Path:
    if subset not in SOURCE_FILENAMES:
        raise ValueError(f"subset must be one of {sorted(SOURCE_FILENAMES)}, got {subset!r}")
    return write_gym_jsonl(
        _iter_source_records(data_dir, subset),
        output=output,
        benchmark_name=BENCHMARK_NAME,
        limit=limit,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        "-d",
        default=HF_DATASET,
        help="HF dataset id, local dataset directory, source JSONL, or downloaded dataset directory.",
    )
    parser.add_argument("--output", "-o", type=Path, default=OUTPUT_FPATH)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--subset", choices=sorted(SOURCE_FILENAMES), default="oss_eval")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(data_dir=args.data_dir, output=args.output, limit=args.limit, subset=args.subset)
