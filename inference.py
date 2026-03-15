# -*- coding: utf-8 -*-
"""Single-sample inference entry point for 3DCvT."""

import argparse
import logging
from pathlib import Path

from inference_runtime import (
    InferenceSession,
    PredictionRequest,
    format_prediction_report,
)


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for single-sample inference."""
    parser = argparse.ArgumentParser(description='3DCvT Inference')
    parser.add_argument('--dataset', type=str, default='lrw', choices=['lrw', 'lrw1000'],
                        help='Dataset type (determines vocabulary and defaults)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth file')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video_path', type=str, help='Path to a raw .mp4 file')
    input_group.add_argument('--pkl_path', type=str, help='Path to a preprocessed .pkl sample')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Dataset root used to reconstruct vocabulary (auto-set if omitted)')
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=5, help='Show top K predictions')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA device id to use')
    return parser


def build_request(args: argparse.Namespace) -> PredictionRequest:
    """Resolve the CLI arguments into one input request."""
    if args.video_path:
        input_path = args.video_path
        declared_type = 'video'
    else:
        input_path = args.pkl_path
        declared_type = 'pkl'

    suffix = Path(input_path).suffix.lower()
    if suffix == '.pkl' and declared_type != 'pkl':
        logger.warning(
            'Input path ends with .pkl but was passed via --video_path. '
            'Treating it as a preprocessed sample.'
        )
        return PredictionRequest(input_path=input_path, input_type='pkl')

    if suffix in {'.mp4', '.avi', '.mov', '.mkv', '.webm'} and declared_type != 'video':
        logger.warning(
            'Input path looks like a raw video but was passed via --pkl_path. '
            'Treating it as a raw video.'
        )
        return PredictionRequest(input_path=input_path, input_type='video')

    return PredictionRequest(input_path=input_path, input_type=declared_type)


def main() -> None:
    """Run single-sample inference using the shared runtime module."""
    args = build_parser().parse_args()
    session = InferenceSession(
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        num_classes=args.num_classes,
        gpu=args.gpu,
    )

    request = build_request(args)
    if request.input_type == 'video':
        logger.info('Processing raw video: %s', request.input_path)
    else:
        logger.info('Processing preprocessed sample: %s', request.input_path)
    result = session.predict_request(request, top_k=args.top_k)

    print(format_prediction_report(result))


if __name__ == '__main__':
    main()
