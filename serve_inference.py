# -*- coding: utf-8 -*-
"""HTTP inference service that keeps one 3DCvT model loaded in memory."""

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

from inference_runtime import InferenceSession, PredictionRequest


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class InferenceRequestHandler(BaseHTTPRequestHandler):
    """Serve health checks and prediction requests over HTTP."""

    session: InferenceSession
    default_top_k: int
    default_batch_size: int

    def do_GET(self) -> None:
        """Return service metadata for readiness checks."""
        if self.path != '/health':
            self._write_json({'error': 'Not found'}, status=HTTPStatus.NOT_FOUND)
            return

        payload = {
            'status': 'ok',
            'dataset': self.session.dataset,
            'data_root': str(self.session.data_root),
            'num_classes': self.session.num_classes,
            'vocab_size': len(self.session.vocab),
            'checkpoint': str(self.session.checkpoint_path),
            'device': str(self.session.device),
        }
        self._write_json(payload)

    def do_POST(self) -> None:
        """Handle single-sample and batch prediction requests."""
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._write_json({'error': str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if self.path == '/predict':
                response = self._handle_predict(payload)
            elif self.path == '/predict_batch':
                response = self._handle_predict_batch(payload)
            else:
                self._write_json({'error': 'Not found'}, status=HTTPStatus.NOT_FOUND)
                return
        except Exception as exc:  # noqa: BLE001 - service must return structured errors.
            self._write_json({'error': str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._write_json(response)

    def log_message(self, format: str, *args) -> None:
        """Route HTTP logs through the project logger."""
        logger.info('%s - %s', self.address_string(), format % args)

    def _handle_predict(self, payload: dict) -> dict:
        """Run inference for a single input."""
        request = self._request_from_payload(payload)
        top_k = int(payload.get('top_k', self.default_top_k))
        result = self.session.predict_request(request, top_k=top_k)
        return result.to_dict()

    def _handle_predict_batch(self, payload: dict) -> dict:
        """Run inference for a list of inputs in one process."""
        items = payload.get('items')
        if not isinstance(items, list) or not items:
            raise ValueError("'items' must be a non-empty list")

        top_k = int(payload.get('top_k', self.default_top_k))
        batch_size = int(payload.get('batch_size', self.default_batch_size))
        requests = [self._request_from_payload(item) for item in items]
        results, failures = self.session.predict_batch(
            requests=requests,
            top_k=top_k,
            batch_size=batch_size,
        )
        return {
            'results': [result.to_dict() for result in results],
            'failures': [failure.to_dict() for failure in failures],
        }

    def _request_from_payload(self, payload: dict) -> PredictionRequest:
        """Normalize the accepted JSON payload formats into one request object."""
        if not isinstance(payload, dict):
            raise ValueError('Request body must be a JSON object')

        video_path = payload.get('video_path')
        pkl_path = payload.get('pkl_path')
        input_path = payload.get('input_path')
        input_type = payload.get('input_type', 'auto')

        provided = [value for value in (video_path, pkl_path, input_path) if value is not None]
        if len(provided) != 1:
            raise ValueError('Provide exactly one of video_path, pkl_path, or input_path.')

        if video_path is not None:
            return PredictionRequest(input_path=str(video_path), input_type='video')
        if pkl_path is not None:
            return PredictionRequest(input_path=str(pkl_path), input_type='pkl')
        return PredictionRequest(input_path=str(input_path), input_type=str(input_type))

    def _read_json(self) -> dict:
        """Read and decode one JSON request body."""
        content_length = int(self.headers.get('Content-Length', '0'))
        if content_length <= 0:
            raise ValueError('Request body is empty')
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON: {exc.msg}') from exc

    def _write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Serialize a JSON response with the given status code."""
        encoded = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the HTTP inference service."""
    parser = argparse.ArgumentParser(description='3DCvT Inference Service')
    parser.add_argument('--dataset', type=str, default='lrw', choices=['lrw', 'lrw1000'],
                        help='Dataset type (determines vocabulary and defaults)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Dataset root used to reconstruct vocabulary (auto-set if omitted)')
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--gpu', type=str, default='0', help='CUDA device id to use')
    parser.add_argument('--top_k', type=int, default=5, help='Default top K predictions to return')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Default batch size for the /predict_batch endpoint')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Service host')
    parser.add_argument('--port', type=int, default=8000, help='Service port')
    return parser


def main() -> None:
    """Start the HTTP inference service."""
    args = build_parser().parse_args()
    session = InferenceSession(
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        num_classes=args.num_classes,
        gpu=args.gpu,
    )

    handler_cls = InferenceRequestHandler
    handler_cls.session = session
    handler_cls.default_top_k = args.top_k
    handler_cls.default_batch_size = args.batch_size

    server = HTTPServer((args.host, args.port), handler_cls)
    logger.info('Inference service listening on http://%s:%s', args.host, args.port)
    logger.info('Health endpoint: GET /health')
    logger.info('Prediction endpoints: POST /predict and POST /predict_batch')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('Shutting down inference service...')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
