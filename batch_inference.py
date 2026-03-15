# -*- coding: utf-8 -*-
"""Batch inference script that reuses one model instance for many inputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from inference_runtime import InferenceSession, PredictionRequest


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for batch inference."""
    parser = argparse.ArgumentParser(description='3DCvT Batch Inference')
    parser.add_argument('--dataset', type=str, default='lrw', choices=['lrw', 'lrw1000'],
                        help='Dataset type (determines vocabulary and defaults)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Dataset root used to reconstruct vocabulary (auto-set if omitted)')
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--gpu', type=str, default='0', help='CUDA device id to use')
    parser.add_argument('--top_k', type=int, default=5, help='Show top K predictions')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of preprocessed samples per forward pass')
    parser.add_argument('--input_type', type=str, default='auto', choices=['auto', 'video', 'pkl'],
                        help='Explicitly set the input type for all paths when auto-detection is not desired')
    parser.add_argument('--input_path', action='append', default=[],
                        help='One input path. Repeat this option to add multiple inputs.')
    parser.add_argument('--input_list', type=str, default=None,
                        help='Text file with one input path per line. Empty lines and # comments are ignored.')
    parser.add_argument('--output', type=str, default='batch_predictions.jsonl',
                        help='Output JSONL path')
    return parser


def load_input_paths(cli_paths: List[str], input_list: Optional[str]) -> List[str]:
    """Merge inline input paths with a newline-delimited input list file."""
    paths = list(cli_paths)
    if input_list is not None:
        list_path = Path(input_list)
        for raw_line in list_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            paths.append(line)
    return paths


def write_results(output_path: str | Path, records: List[dict]) -> None:
    """Write one JSON object per line for downstream processing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main() -> None:
    """Run batch inference in one long-lived Python process."""
    args = build_parser().parse_args()
    input_paths = load_input_paths(args.input_path, args.input_list)
    if not input_paths:
        raise SystemExit('No inputs were provided. Use --input_path or --input_list.')

    session = InferenceSession(
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        num_classes=args.num_classes,
        gpu=args.gpu,
    )

    requests = [
        PredictionRequest(input_path=path, input_type=args.input_type)
        for path in input_paths
    ]
    logger.info('Running batch inference for %s inputs...', len(requests))
    results, failures = session.predict_batch(
        requests=requests,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    result_map = {result.input_path: result.to_dict() for result in results}
    failure_map = {failure.input_path: failure.to_dict() for failure in failures}
    ordered_records = []
    for path in input_paths:
        if path in result_map:
            ordered_records.append(result_map[path])
        else:
            ordered_records.append(failure_map[path])

    write_results(args.output, ordered_records)
    logger.info('Saved %s records to %s', len(ordered_records), args.output)
    if failures:
        logger.warning('Batch finished with %s failed inputs.', len(failures))


if __name__ == '__main__':
    main()
