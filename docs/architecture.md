# OpenVLM Architecture

This document provides an overview of the OpenVLM framework's architecture. Our goal is to create a modular, extensible, and efficient system for post-training and deploying Vision-Language Models (VLMs) for engineering applications.

## Core Philosophy

- **Modularity:** Components are designed to be loosely coupled, allowing for easy replacement or extension.
- **Configuration-Driven:** Operations are primarily controlled via YAML configuration files, promoting reproducibility and simplifying experimentation.
- **Extensibility:** The framework is built to accommodate new models, training techniques, data types, and integrations.
- **Efficiency:** Performance is a key consideration, with support for distributed training, mixed-precision, and C++ extensions where beneficial.

## High-Level Diagram

(A visual diagram will be added here in a future update. For now, imagine a layered architecture.)

```mermaid
graph TD
    A[User/CLI/SDK] --> B{VLMEngineer Core Agent};

    B --> C{Configuration Module};
    C -->|Loads| D[YAML Config Files];

    B --> E{Data Handling Module};
    E --> F[Dataset Loaders];
    E --> G[Data Augmentation];
    E --> H[Data Preprocessing];
    F --> I[Image Data];
    F --> J[Text/JSONL Data];

    B --> K{Model Management Module};
    K --> L[LLM Wrappers (e.g., Transformers)];
    K --> M[Vision Encoders (e.g., CLIP)];
    K --> N[Fusion/Adapter Layers];
    N --> O[LoRA, QLoRA, etc.];

    B --> P{Training & Optimization Module};
    P --> Q[Training Loops (SFT, Quant, etc.)];
    P --> R[Optimizers (AdamW, etc.)];
    P --> S[Schedulers];
    P --> T[Loss Functions];
    P --> U[Accelerate/Distributed Training];

    B --> V{Evaluation Module};
    V --> W[Metrics Computation];
    V --> X[Benchmark Datasets];

    B --> Y{Inference Engine};
    Y --> Z[Generation/Decoding Strategies];

    B --> AA{Integration Layer};
    AA --> BB[Datasette];
    AA --> CC[LLM CLI Tool];
    AA --> DD[sqlite-utils];
    AA --> EE[CAD/Simulation Tools (Future)];

    subgraph "Core Library (open_vlm/)"
        direction LR
        C; E; K; P; V; Y; AA;
    end

    subgraph "External Interactions"
        A; D; I; J; X; BB; CC; DD; EE;
    end
```

## Key Components (Directory Structure Mapping)

This maps to the `open_vlm/` directory structure:

*   **`open_vlm/`**
    *   **`__init__.py`**: Package initializer.
    *   **`api/`** (Future): For serving models via REST APIs (e.g., using FastAPI).
        *   `endpoints.py`
        *   `schemas.py`
    *   **`cli/`**: Command-Line Interface logic.
        *   `main.py`: Entry point for the `open-vlm` command.
        *   `train_cmd.py`: Handles `open-vlm train ...` commands.
        *   `infer_cmd.py`: Handles `open-vlm infer ...` commands.
        *   (Other command group files)
    *   **`config/`**: Configuration handling.
        *   `core_config.py`: Base configuration dataclasses (e.g., `ModelConfig`, `DataConfig`, `TrainingConfig`).
        *   `vision_sft_config.py`: Specific `VisionSFTConfig` dataclass, inheriting from base configs.
        *   `quantitative_config.py`: Specific `QuantitativeReasoningConfig`.
        *   `parser.py`: Logic to load and validate YAML configs into dataclass instances.
    *   **`core/`**: The heart of the VLM engineering capabilities.
        *   `VLMEngineer.py`: The main orchestrator class. It takes a configuration object and manages the overall workflow (training, inference, evaluation).
        *   **`adapters/`**: Implementation of various model adaptation techniques.
            *   `lora.py`: LoRA and QLoRA implementations.
            *   `base_adapter.py`: Abstract adapter class.
        *   **`datasets/`**: Data loading, preprocessing, and augmentation.
            *   `base_dataset.py`: Base PyTorch `Dataset` class for VLMs.
            *   `sft_dataset.py`: Dataset class for Supervised Fine-Tuning.
            *   `image_processing.py`: Image transformation and augmentation functions.
            *   `text_tokenization.py`: Text tokenization utilities, wrapping Hugging Face tokenizers.
        *   **`training/`**: Training loops, optimization, and related utilities.
            *   `trainer.py`: Core training loop logic, potentially using Hugging Face `Trainer` or a custom loop integrated with `accelerate`.
            *   `losses.py`: Custom loss functions if needed.
            *   `optimization.py`: Optimizer and scheduler setup.
        *   **`vision/`**: Integration with vision models/encoders.
            *   `clip_wrapper.py`: Wrapper for CLIP models.
            *   `base_vision_tower.py`: Abstract class for vision backbones.
    *   **`integration/`**: Modules for integrating with external tools and libraries.
        *   `datasette_utils.py`: Functions to export VLM outputs (e.g., structured data from diagrams) to SQLite databases for Datasette.
        *   `llm_tool_utils.py`: Utilities to interact with the `llm` CLI tool (e.g., for further processing of generated text).
        *   `sqlite_utils_integration.py`: Using `sqlite-utils` for more advanced database interactions.
    *   **`models/`**: VLM model definitions, architectures, and fusion strategies.
        *   `base_vlm.py`: Abstract base VLM class.
        *   `openvlm_model.py`: The main VLM architecture, combining LLM, vision encoder, and projection/fusion layers.
        *   `projectors.py`: Different types of projection layers (e.g., MLP, Q-Former like).
    *   **`templates/`** (Future): Project or component templates for quickly starting new VLM tasks.
    *   **`tests/`**: Unit, integration, and functional tests.
        *   `test_config.py`
        *   `test_datasets.py`
        *   `test_training_smoke.py` (smoke tests for training runs)
        *   `test_cli.py`
    *   **`utils/`**: Common utility functions and helper classes.
        *   `logging_utils.py`: Setup for consistent logging.
        *   `checkpoint_utils.py`: Saving and loading model checkpoints.
        *   `distributed_utils.py`: Helpers for distributed training setups.

## Workflow Example: Vision SFT Training via CLI

1.  **User Input**: `open-vlm train vision-sft --config configs/vision_sft_config.yaml`
2.  **CLI (`cli/main.py`, `cli/train_cmd.py`)**: Parses the command and identifies the task and config file.
3.  **Configuration (`config/parser.py`, `config/vision_sft_config.py`)**: Loads `vision_sft_config.yaml` into a `VisionSFTConfig` dataclass instance.
4.  **Core Agent (`core/VLMEngineer.py`)**: `VLMEngineer` is instantiated with the `VisionSFTConfig`.
5.  **Data Handling (`core/datasets/`)**: The `VLMEngineer` initializes the SFT dataset (`sft_dataset.py`) using paths and parameters from the config. This involves:
    *   Reading JSONL files.
    *   Setting up image loading (`image_processing.py`).
    *   Setting up text tokenization (`text_tokenization.py`).
6.  **Model Loading (`models/`, `core/vision/`, `core/adapters/`)**: The `VLMEngineer` loads/initializes:
    *   The specified LLM (e.g., Llama-2 from Hugging Face).
    *   The specified vision encoder (e.g., CLIP from Hugging Face).
    *   Projection layers (`models/projectors.py`).
    *   If `use_lora` is true, LoRA layers are applied (`core/adapters/lora.py`).
7.  **Training (`core/training/`)**: The `VLMEngineer` initiates the training process:
    *   A `Trainer` (or custom training loop) is configured with the model, datasets, optimizer, scheduler, and loss function.
    *   If distributed training is enabled, `accelerate` (or similar) handles the distribution.
    *   The training loop iterates over the data, performs forward/backward passes, and updates model weights.
    *   Logging (W&B, console) and checkpointing (`utils/checkpoint_utils.py`) occur periodically.
8.  **Output**: Trained model artifacts (weights, adapter configs), logs, and evaluation results are saved to the specified `output_dir`.

## Future Directions Mentioned in Roadmap

*   **Datasette/LLM/sqlite-utils Integration (`integration/`)**: These modules will house the logic to convert VLM outputs (like structured data extracted from diagrams or text analyses) into SQLite databases. They will also facilitate using `llm` for further processing of generated text and `sqlite-utils` for flexible database operations.
*   **GUI Interaction Module**: This might involve a new sub-package, possibly `core/gui_interaction/`, with components for screen capture, UI element detection (potentially using the vision model itself or specialized models), and action generation.
*   **Advanced Reasoning (Causal, Physics-Informed)**: These would likely be new modules within `core/` or `models/`, potentially introducing new model architectures or training objectives.

This architecture is a living design and will evolve as OpenVLM grows. Community feedback and contributions are crucial to its refinement. 