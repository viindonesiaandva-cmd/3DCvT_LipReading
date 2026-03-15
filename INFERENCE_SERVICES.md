# Inference, Batch Inference, and Online Service Guide

This document explains how to use the inference-related scripts in this repository:

- `inference.py`: single-sample inference
- `batch_inference.py`: batch inference in one long-lived Python process
- `serve_inference.py`: online HTTP inference service with one model instance kept in memory

The goal is to avoid repeated Python startup and repeated checkpoint loading when you need to run many samples.

## 1. Script Overview

| Script | Use Case | Model Loading Behavior | Typical Scenario |
| --- | --- | --- | --- |
| `inference.py` | One sample at a time | Loads the model once per command | Quick manual check |
| `batch_inference.py` | Many local files | Loads the model once for the whole batch job | Offline inference over a list of files |
| `serve_inference.py` | Repeated requests from other processes | Loads the model once and stays resident | Local API service / demo backend |

## 2. Input Types

The inference runtime supports two input types:

- Raw video files: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Preprocessed samples: `.pkl`

Recommended usage:

- `LRW`: usually use raw `.mp4` files
- `LRW-1000`: usually use preprocessed `.pkl` files

Although `inference.py` now auto-corrects obvious flag mistakes, you should still use the correct flag:

- `--video_path` for raw videos
- `--pkl_path` for preprocessed samples

## 3. Environment Requirements

Use the project environment, not the system Python. For example:

```bash
conda activate 3DCvT
python --version
```

If you accidentally use a Python interpreter outside the project environment, you may see errors like:

```text
ModuleNotFoundError: No module named 'einops'
```

That is an environment problem, not an inference-script problem.

## 4. Single-Sample Inference

### 4.1 LRW Raw Video

```bash
python inference.py \
  --dataset lrw \
  --video_path /path/to/sample.mp4 \
  --checkpoint experiments/3DCvT_LRW_new_version/ckpts/best_model.pth \
  --gpu 0
```

### 4.2 LRW-1000 PKL Sample

```bash
python inference.py \
  --dataset lrw1000 \
  --pkl_path /path/to/sample.pkl \
  --checkpoint experiments/3DCvT_LRW1000_new_version/ckpts/best_model.pth \
  --gpu 0
```

### 4.3 Auto-Correction for Wrong Flag

If you pass a `.pkl` file through `--video_path`, the script will warn and automatically treat it as a PKL sample:

```bash
python inference.py \
  --dataset lrw1000 \
  --video_path /path/to/sample.pkl \
  --checkpoint experiments/3DCvT_LRW1000_new_version/ckpts/best_model.pth \
  --gpu 0
```

Expected behavior:

- The script prints a warning
- The request is internally converted to PKL inference
- Inference still runs normally

## 5. Batch Inference

`batch_inference.py` is designed for offline jobs where you have many inputs and do not want to start a fresh Python process for every file.

### 5.1 Repeated `--input_path`

```bash
python batch_inference.py \
  --dataset lrw \
  --checkpoint experiments/3DCvT_LRW_new_version/ckpts/best_model.pth \
  --gpu 0 \
  --batch_size 8 \
  --input_path /path/to/A.mp4 \
  --input_path /path/to/B.mp4 \
  --output batch_predictions.jsonl
```

### 5.2 `--input_list`

Create a text file with one input path per line:

```text
/path/to/A.mp4
/path/to/B.mp4
/path/to/C.mp4
```

Then run:

```bash
python batch_inference.py \
  --dataset lrw \
  --checkpoint experiments/3DCvT_LRW_new_version/ckpts/best_model.pth \
  --gpu 0 \
  --batch_size 8 \
  --input_list /path/to/inputs.txt \
  --output batch_predictions.jsonl
```

### 5.3 LRW-1000 Batch Example

```bash
python batch_inference.py \
  --dataset lrw1000 \
  --checkpoint experiments/3DCvT_LRW1000_new_version/ckpts/best_model.pth \
  --gpu 0 \
  --batch_size 8 \
  --input_list /path/to/lrw1000_inputs.txt \
  --output lrw1000_predictions.jsonl
```

### 5.4 Important Arguments

| Argument | Meaning |
| --- | --- |
| `--dataset` | `lrw` or `lrw1000` |
| `--checkpoint` | Checkpoint path |
| `--gpu` | GPU id, such as `0` or `1` |
| `--batch_size` | Number of prepared samples per model forward pass |
| `--input_type` | `auto`, `video`, or `pkl` |
| `--input_path` | One input path; can be repeated |
| `--input_list` | Text file containing one path per line |
| `--output` | JSONL output path |

### 5.5 Output Format

Output is written as JSONL: one JSON object per line.

Successful record example:

```json
{"input_path":"/path/to/A.mp4","input_type":"video","top_prediction":"CAMERON","top_confidence":60.30,"predictions":[{"rank":1,"word":"CAMERON","confidence":60.30}]}
```

Failure record example:

```json
{"input_path":"/path/to/missing.mp4","input_type":"video","error":"Cannot open video: /path/to/missing.mp4"}
```

This means a batch job can complete even when part of the input list is bad.

## 6. Online Inference Service

`serve_inference.py` starts a local HTTP service. The model is loaded once during startup and reused for all requests.

### 6.1 Start the Service

LRW example:

```bash
python serve_inference.py \
  --dataset lrw \
  --checkpoint experiments/3DCvT_LRW_new_version/ckpts/best_model.pth \
  --gpu 0 \
  --host 127.0.0.1 \
  --port 8000
```

LRW-1000 example:

```bash
python serve_inference.py \
  --dataset lrw1000 \
  --checkpoint experiments/3DCvT_LRW1000_new_version/ckpts/best_model.pth \
  --gpu 0 \
  --host 127.0.0.1 \
  --port 8001
```

### 6.2 Health Check

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "dataset": "lrw",
  "data_root": "/path/to/data_LRW",
  "num_classes": 500,
  "vocab_size": 500,
  "checkpoint": "/path/to/best_model.pth",
  "device": "cuda:0"
}
```

### 6.3 Single Prediction Endpoint

Endpoint:

```text
POST /predict
```

Raw video request:

```bash
curl --noproxy '*' \
  -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"video_path": "/path/to/sample.mp4", "top_k": 5}'
```

PKL request:

```bash
curl --noproxy '*' \
  -X POST http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"pkl_path": "/path/to/sample.pkl", "top_k": 5}'
```

Alternative generic request form:

```bash
curl --noproxy '*' \
  -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_path": "/path/to/sample.mp4", "input_type": "video", "top_k": 5}'
```

Response example:

```json
{
  "input_path": "/path/to/sample.mp4",
  "input_type": "video",
  "top_prediction": "CAMERON",
  "top_confidence": 60.30,
  "predictions": [
    {"rank": 1, "word": "CAMERON", "confidence": 60.30},
    {"rank": 2, "word": "HAPPEN", "confidence": 6.56}
  ]
}
```

### 6.4 Batch Prediction Endpoint

Endpoint:

```text
POST /predict_batch
```

Example request:

```bash
curl --noproxy '*' \
  -X POST http://127.0.0.1:8000/predict_batch \
  -H 'Content-Type: application/json' \
  -d '{
        "batch_size": 4,
        "top_k": 3,
        "items": [
          {"video_path": "/path/to/A.mp4"},
          {"video_path": "/path/to/B.mp4"}
        ]
      }'
```

Response example:

```json
{
  "results": [
    {
      "input_path": "/path/to/A.mp4",
      "input_type": "video",
      "top_prediction": "CAMERON",
      "top_confidence": 60.30,
      "predictions": [
        {"rank": 1, "word": "CAMERON", "confidence": 60.30}
      ]
    }
  ],
  "failures": [
    {
      "input_path": "/path/to/missing.mp4",
      "input_type": "video",
      "error": "Cannot open video: /path/to/missing.mp4"
    }
  ]
}
```

## 7. Request Rules

For `/predict`:

- Provide exactly one of:
  - `video_path`
  - `pkl_path`
  - `input_path`

For `/predict_batch`:

- Provide `items` as a non-empty list
- Each item must follow the same one-input rule as `/predict`

If the request is invalid, the service returns HTTP `400` with a JSON error body.

Example:

```json
{"error": "'items' must be a non-empty list"}
```

## 8. Performance Notes

### 8.1 Why `batch_inference.py` Is Faster Than Repeated `inference.py`

`inference.py` pays the following startup cost every time:

- Python startup
- PyTorch import
- model construction
- checkpoint loading

`batch_inference.py` avoids repeated checkpoint loading within one job.

### 8.2 Why `serve_inference.py` Is Faster For Repeated Requests

`serve_inference.py` keeps the model resident in memory, so later requests avoid:

- process startup
- repeated checkpoint loading

This is the most practical option if you need a local demo server or repeated programmatic access.

### 8.3 Batch Size Guidance

- Start with `--batch_size 4` or `--batch_size 8`
- Increase only if GPU memory allows it
- For LRW-1000, memory pressure may differ depending on your GPU and request mix

## 9. Troubleshooting

### 9.1 `.pkl` Passed to `--video_path`

Symptom:

- You accidentally pass a PKL file using `--video_path`

Current behavior:

- The script warns and auto-corrects to PKL inference

Best practice:

- Still use `--pkl_path` for PKL files

### 9.2 `ModuleNotFoundError: No module named 'einops'`

Cause:

- Wrong Python interpreter

Fix:

```bash
conda activate 3DCvT
python inference.py --help
```

### 9.3 `HTTP Error 502` When Requesting `127.0.0.1`

Cause:

- Local requests are going through a proxy defined in your environment

Fix:

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
```

If you call the service from Python, disable proxies explicitly in the client.

### 9.4 Bad Predictions With a Valid Command

Check the following:

- Are you using the checkpoint that matches the current model code?
- Are you using the correct dataset (`lrw` vs `lrw1000`)?
- Are you reconstructing the vocabulary from the correct `data_root`?

Older checkpoints from incompatible model variants may load with warnings but give unreliable predictions.

## 10. Recommended Workflow

### Case A: Quick sanity check

Use:

- `inference.py`

### Case B: Run a folder or list of files offline

Use:

- `batch_inference.py`

### Case C: Build a local demo, UI, or external caller

Use:

- `serve_inference.py`

## 11. Locally Validated Paths

The following paths were validated on this machine during development:

- LRW checkpoint:
  `experiments/3DCvT_LRW_new_version/ckpts/best_model.pth`
- LRW-1000 checkpoint:
  `experiments/3DCvT_LRW1000_new_version/ckpts/best_model.pth`
- LRW sample:
  `/public/dataset/LRW/lipread_mp4/CAMERON/val/CAMERON_00001.mp4`
- LRW-1000 sample:
  `/ssd2/3DCvT_data/data_LRW1000/案件/train/0ccaa70a95a1f532bbb9cc4aefca6eba_7p94_8p46.pkl`

These concrete paths are only examples from the local machine. Replace them with your own paths when sharing the repository.

## 12. Reproducing the Validation Run

The repository includes a validation harness that reruns the inference-related checks and saves all logs and artifacts under `debug/`.

Script:

```text
debug/run_inference_validation.py
```

Example:

```bash
/home/master_chief117/miniconda3/envs/3DCvT/bin/python \
  debug/run_inference_validation.py \
  --sample_count 5
```

Meaning of `--sample_count`:

- For LRW, the script picks that many raw validation videos
- For LRW-1000, the script picks that many PKL samples
- It also adds one LRW-1000 flag-mismatch case on purpose to verify the `.pkl` vs `--video_path` auto-correction path

What the script does:

- Parses the training logs to recover the processed `data_root`
- Confirms the project Python environment and script syntax
- Runs `--help` checks for all three inference entry points
- Runs single-sample inference for multiple LRW and LRW-1000 examples
- Runs batch inference for LRW and LRW-1000
- Starts both HTTP services locally and verifies:
  - `GET /health`
  - `POST /predict`
  - `POST /predict_batch`
  - one invalid request path that must return HTTP `400`

## 13. Validation Artifact Layout

Each run creates a timestamped directory:

```text
debug/inference_validation/<timestamp>/
```

There is also a moving shortcut:

```text
debug/inference_validation/latest
```

Typical layout:

```text
debug/inference_validation/<timestamp>/
├── SUMMARY.md
├── manifest.json
├── commands/
├── samples/
└── artifacts/
    ├── single/
    ├── batch/
    └── service/
```

### 13.1 Top-Level Files

`SUMMARY.md`

- Human-readable summary of the whole run
- Includes data roots, checkpoints, sample lists, and command log paths

`manifest.json`

- Machine-readable manifest
- Contains the exact commands that were executed
- Records the generated artifact paths

### 13.2 `commands/`

Contains the logs for command-level checks such as:

- `py_compile`
- `inference.py --help`
- `batch_inference.py --help`
- `serve_inference.py --help`

Each log records:

- start time
- end time
- exit code
- full command
- stdout
- stderr

### 13.3 `samples/`

Contains the exact sample lists used for that validation run.

Examples:

- `lrw_samples.txt`
- `lrw1000_samples.txt`
- `lrw_batch_inputs.txt`
- `lrw1000_batch_inputs.txt`

This means you can always tell exactly which files were used for the outputs in the same run directory.

### 13.4 `artifacts/single/`

Contains one log file per single-sample inference command.

Examples:

- `lrw_single_01.log`
- `lrw1000_single_03.log`
- `lrw1000_single_flag_mismatch.log`

Also contains:

- `single_summary.json`

`single_summary.json` is the fastest place to inspect the overall effect of the single-sample tests. It records:

- input path
- test type
- exit code
- extracted top prediction
- the log file path for the full raw output

### 13.5 `artifacts/batch/`

Contains:

- batch command logs
- batch JSONL outputs

Examples:

- `lrw_batch.log`
- `lrw_batch_predictions.jsonl`
- `lrw1000_batch.log`
- `lrw1000_batch_predictions.jsonl`

The JSONL outputs contain:

- one JSON object per successful prediction
- or one JSON object with an `error` field for failed inputs

This is useful because it lets you verify that a batch job can continue even if one input file is bad.

### 13.6 `artifacts/service/`

Contains:

- service startup and request logs
- serialized HTTP responses for each endpoint check

Examples:

- `lrw_service.log`
- `lrw_health.json`
- `lrw_predict.json`
- `lrw_predict_batch.json`
- `lrw1000_service.log`
- `lrw1000_health.json`
- `lrw1000_predict.json`
- `lrw1000_predict_batch.json`
- `lrw1000_predict_batch_invalid.json`

The `*_invalid.json` files are intentional negative tests. They show what the service returns when the request payload is malformed.

## 14. How To Review the Results Quickly

If you want the shortest review path after a validation run, use this order:

1. Open `debug/inference_validation/latest/SUMMARY.md`
2. Open `debug/inference_validation/latest/artifacts/single/single_summary.json`
3. Open the batch JSONL files
4. Open the service `*.json` responses
5. Only if needed, drill into the individual `.log` files

That order gives you the high-level behavior first, then the raw logs only when something looks wrong.
