# OpenVLM CLI Usage Guide

This document provides instructions and examples for using the OpenVLM Command Line Interface (CLI).

## Base Command

The base command for all CLI operations is `open-vlm`:

```bash
open-vlm [COMMAND] [SUBCOMMAND] [OPTIONS]
```

## Main Commands

### 1. `train`

Train or fine-tune models.

**Options:**
- `--config PATH`: Path to the YAML configuration file
- `--output-dir DIRECTORY`: Override output directory
- `--num-gpus INT`: Override number of GPUs

**Example:**
```bash
open-vlm train --config configs/vision_sft_config.yaml
```

### 2. `infer`

Run inference with a trained model.

**Options:**
- `--model-path PATH`: Path to trained model
- `--image-path PATH`: Input image path
- `--prompt TEXT`: Text prompt
- `--max-tokens INT`: Max tokens to generate
- `--output-file PATH`: Save output to file

**Example:**
```bash
open-vlm infer \
    --model-path ./output/vision_sft_experiment \
    --image-path ./examples/sample_diagram.png \
    --prompt "Describe this technical diagram"
```

### 3. `serve`

Launch REST API server.

**Options:**
- `--model-path PATH`: Model path
- `--host TEXT`: Host address
- `--port INT`: Port number
- `--workers INT`: Worker processes
- `--reload`: Enable auto-reload
- `--cors`: Enable CORS

**Example:**
```bash
open-vlm serve --model-path ./output/vision_sft_experiment --port 8000
```

### 4. `export`

Export results to various formats.

**Options:**
- `results_file`: Input results file
- `--output PATH`: Output file path
- `--table TEXT`: Table name for SQLite
- `--metadata PATH`: Metadata JSON file
- `--format FORMAT`: Output format (sqlite/json/csv)

**Example:**
```bash
open-vlm export inference_results.json --output analysis_results.db
```

## Global Options

- `--version`: Display version
- `--help`: Show help
- `-v`: Verbose output

## API Endpoints

The server provides these endpoints:
- `GET /health`: Health check
- `GET /model-info`: Model information
- `POST /analyze`: Analyze image with prompt
- `POST /upload`: Upload image for analysis
- `GET /export-analytics`: Export analytics

## Configuration

Configuration files in YAML format control training parameters. See `configs/` for examples. 