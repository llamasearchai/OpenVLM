# OpenVLM Installation Guide

This guide provides detailed instructions for installing OpenVLM and its dependencies.

## Prerequisites

- Python 3.8 or higher
- Pip package manager
- (Optional but Recommended) Conda or a virtual environment manager

## Installation Options

Choose the installation method that best suits your needs:

### 1. Full Installation (Recommended)

This installs OpenVLM with all optional features, including spatial reasoning and diagram analysis capabilities, along with development tools.

```bash
pip install "open-vlm[spatial,diagram,dev]"
```

This single command handles the installation of:
- Core `open-vlm` package
- Dependencies for spatial reasoning (`pyrender`, `trimesh`)
- Dependencies for diagram analysis (`pytesseract`, `easyocr`)
- Development tools (`pytest`, `black`, `isort`)

### 2. Basic Installation

Installs only the core `open-vlm` package.

```bash
pip install open-vlm
```

### 3. Installation with Specific Extras

You can install support for specific features:

- **Spatial Reasoning:**
  ```bash
  pip install "open-vlm[spatial]"
  ```
- **Diagram Analysis:**
  ```bash
  pip install "open-vlm[diagram]"
  ```
- **Development Tools:**
  ```bash
  pip install "open-vlm[dev]"
  ```

You can combine extras, for example: `pip install "open-vlm[spatial,diagram]"`

## Verifying Installation

After installation, you can verify it by importing `open_vlm` in a Python interpreter or by running the CLI:

```bash
open-vlm --version
```
(Note: `--version` flag to be implemented in the CLI)

## Troubleshooting

**Common Issues:**

*   **Dependency Conflicts:** If you encounter dependency conflicts, try creating a fresh virtual environment:
    ```bash
    conda create -n openvlm_env python=3.9 -y
    conda activate openvlm_env
    pip install "open-vlm[spatial,diagram,dev]"
    ```
*   **PyTorch CUDA Issues:** Ensure your PyTorch installation matches your CUDA version if you plan to use GPUs. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
*   **Tesseract OCR Engine (for diagram analysis):** `pytesseract` requires the Tesseract OCR engine to be installed on your system.
    *   **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
    *   **macOS (using Homebrew):** `brew install tesseract`
    *   **Windows:** Download the installer from the [Tesseract GitHub page](https://github.com/UB-Mannheim/tesseract/wiki). Ensure Tesseract is added to your system's PATH.

**Reporting Issues:**
If you encounter any problems not covered here, please [open an issue](https://github.com/jina-ai/open-vlm/issues) on our GitHub repository, providing as much detail as possible about your system and the error. 