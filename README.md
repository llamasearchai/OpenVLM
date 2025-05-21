<p align="center">
  <img src="OpenVLM.svg" alt="OpenVLM Logo" width="200"/>
</p>

<h1 align="center">OpenVLM: Pioneering the Future of Engineering with Vision-Language Intelligence</h1>

<p align="center">
  <a href="https://github.com/jina-ai/open-vlm/actions/workflows/ci.yml">
    <img src="https://github.com/jina-ai/open-vlm/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/open-vlm/">
    <img src="https://img.shields.io/pypi/v/open-vlm.svg" alt="PyPI Version">
  </a>
  <a href="https://github.com/jina-ai/open-vlm/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/open-vlm.svg" alt="License">
  </a>
  <a href="https://github.com/jina-ai/open-vlm/stargazers">
    <img src="https://img.shields.io/github/stars/jina-ai/open-vlm.svg" alt="GitHub Stars">
  </a>
  <a href="https://github.com/jina-ai/open-vlm/issues">
    <img src="https://img.shields.io/github/issues/jina-ai/open-vlm.svg" alt="GitHub Issues">
  </a>
  <a href="https://twitter.com/intent/follow?screen_name=jinaai">
    <img src="https://img.shields.io/twitter/follow/jinaai?style=social&logo=twitter" alt="Follow Jina AI on Twitter">
  </a>
</p>

## Overview

OpenVLM is an advanced open-source framework designed to empower AI-driven engineering through vision-language intelligence. The framework bridges complex visual data with sophisticated language understanding, enabling precise analysis and interpretation of engineering problems.

## Key Features

- **Advanced VLM Post-Training**: Specialized techniques for fine-tuning leading VLMs (Llama-2, CLIP) for engineering applications
- **Quantitative Reasoning Engine**: Dedicated module for processing numerical data in visual information
- **Technical Diagram Understanding**: Algorithms for parsing engineering drawings, P&IDs, and circuit diagrams
- **Spatial Reasoning**: Capabilities for understanding spatial relationships and 3D geometries
- **GUI Interaction**: Framework for training agents to interact with engineering software interfaces
- **Engineering Data Augmentation**: Tailored strategies for engineering datasets
- **Comprehensive Evaluation**: Tools for measuring VLM performance on engineering tasks
- **Scalable Architecture**: Support for distributed training and optimized components
- **Modular Design**: Easy integration with existing engineering tools and future VLM advancements

## Installation

```bash
# Full installation with all capabilities
pip install "open-vlm[spatial,diagram,dev]"

# Basic installation
pip install open-vlm

# With spatial reasoning support
pip install "open-vlm[spatial]"

# With diagram analysis support
pip install "open-vlm[diagram]"

# For developers including testing tools
pip install "open-vlm[dev]"
```

## Quick Start

### Basic Usage: Vision Supervised Fine-Tuning

```python
from open_vlm.config import VisionSFTConfig, TrainingMethod
from open_vlm.core import VLMEngineer

config = VisionSFTConfig(
    method=TrainingMethod.VISION_SFT,
    model_name="meta-llama/Llama-2-7b-hf",
    vision_model_name="openai/clip-vit-large-patch14",
    learning_rate=2e-5,
    batch_size=8,
    num_epochs=3,
    output_dir="./output/vision_sft_experiment",
    train_file="./data/mock_train.jsonl",
    eval_file="./data/mock_eval.jsonl",
    image_dir="./data/mock_images",
    use_lora=True
)

agent = VLMEngineer(config)
model = agent.run()
```

### Command Line Interface

```bash
# Train a model
open-vlm train --config configs/vision_sft_config.yaml

# Run inference
open-vlm infer --model-path ./output/vision_sft_experiment --image-path diagram.png --prompt "Describe this technical diagram"

# Launch API server
open-vlm serve --model-path ./output/vision_sft_experiment --port 8000
```

## Architecture Overview

OpenVLM is built upon a modular architecture to ensure flexibility and scalability:

```
OpenVLM/
├── configs/                     # YAML configuration files for training and inference
├── data/                        # (User-provided) Example datasets and image directories
│   ├── mock_train.jsonl
│   ├── mock_eval.jsonl
│   └── mock_images/
├── docs/                        # Detailed documentation
│   ├── installation.md
│   ├── data_preparation.md
│   ├── cli_usage.md
│   └── architecture.md
├── examples/                    # Practical examples and use-cases
│   └── basic_usage.py
├── open_vlm/                    # Core library code
│   ├── __init__.py
│   ├── api/                     # API endpoints
│   ├── cli/                     # Command-line interface logic
│   ├── config/                  # Configuration dataclasses and parsers
│   ├── core/                    # Main VLM engineering components
│   │   ├── adapters/            # Model adaptation layers (e.g., LoRA)
│   │   ├── datasets/            # Data loading and preprocessing
│   │   ├── training/            # Training loops and optimization
│   │   └── vision/              # Vision model integration
│   ├── integration/             # Integrations with external tools (Datasette, LLM, etc.)
│   ├── models/                  # VLM model definitions and architectures
│   ├── templates/               # Project templates (future)
│   ├── tests/                   # Unit and integration tests
│   └── utils/                   # Utility functions and helper classes
├── output/                      # Default directory for trained models and results
├── OpenVLM.svg                  # Project logo
├── README.md                    # This file
├── setup.py                     # Package setup script
├── LICENSE
└── CONTRIBUTING.md
```
A more detailed architectural diagram and explanation will be available in [`docs/architecture.md`](./docs/architecture.md).

## Configuration

OpenVLM uses YAML configuration files (see `configs/` directory) to define training, inference, and other operational parameters. This allows for easy management and reproducibility of experiments.

Key configuration aspects include:
*   Model selection (LLM and vision encoder)
*   Training hyperparameters (learning rate, batch size, epochs)
*   Dataset paths and formats
*   LoRA and quantization settings
*   Logging and output directories

Refer to `configs/vision_sft_config.yaml` for a comprehensive example.

## Contributing

We welcome contributions from the community! Whether it's bug fixes, new features, improved documentation, or exciting examples, your help is invaluable.

Please check out our [Contribution Guidelines](CONTRIBUTING.md) for details on how to get started, our coding standards, and the development process. We also encourage you to open an [Issue](https://github.com/jina-ai/open-vlm/issues) to discuss any proposed changes or to report bugs.

## Community & Support

*   **GitHub Discussions:** For questions, feature requests, and general discussions.
*   **Issue Tracker:** Report bugs and track their status.
*   **(Future) Discord Server:** For real-time community interaction and support.

## Citation

If you use OpenVLM in your research or projects, please cite us (details to be added once a paper/preprint is available). For now, you can cite the repository:

```bibtex
@software{OpenVLM_Framework,
  author       = {{Jina AI and Community Contributors}},
  title        = {{OpenVLM: An Advanced Open-Source Framework for Vision-Language Model Post-Training in Engineering Applications}},
  year         = {2023}, # Or current year
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/jina-ai/open-vlm}}
}
```

## License

OpenVLM is licensed under the [Apache License 2.0](LICENSE).

---

<p align="center">
Empowering the next wave of engineering innovation, together.
</p> 