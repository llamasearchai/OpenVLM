# OpenVLM Data Preparation Guide

This guide outlines the expected data formats and preprocessing steps for using OpenVLM, particularly for Vision Supervised Fine-Tuning (SFT) and other training methodologies.

## General Principles

- **Clear Directory Structure:** Maintain a well-organized directory structure for your datasets and images.
- **JSON Lines (JSONL) Format:** Training and evaluation metadata are primarily expected in JSONL format, where each line is a valid JSON object representing a data sample.
- **Image Paths:** JSONL files should reference images using relative paths from a specified `image_dir` or absolute paths.

## Vision Supervised Fine-Tuning (SFT) Data

For Vision SFT, you typically need:
1.  A training data file (e.g., `train.jsonl`)
2.  An evaluation data file (e.g., `eval.jsonl`)
3.  A directory containing all referenced images.

### JSONL File Format for SFT

Each JSON object in your `.jsonl` file should (at a minimum) contain:

```json
{
  "image_id": "unique_image_identifier_001",
  "image_path": "relative/path/to/image_001.jpg", // Or an absolute path
  "prompt": "Describe the key components in this engineering drawing.",
  "response": "The drawing shows a gear pump assembly with labels for the drive gear, idler gear, inlet port, and outlet port."
}
```

**Key Fields:**

*   `image_id` (str): A unique identifier for the image. Useful for tracking and referencing.
*   `image_path` (str): The path to the image file. This can be relative to the `image_dir` specified in your configuration or an absolute path.
*   `prompt` (str): The input prompt or question related to the image that will be fed to the language model part of the VLM.
*   `response` (str): The desired ground-truth textual response that the model should learn to generate for the given image and prompt.

**Optional Fields (depending on your specific SFT task):**

*   `bounding_boxes` (list of dicts): For tasks requiring object detection or localization. Each dict might contain `{"label": "component_name", "box": [x_min, y_min, x_max, y_max]}`.
*   `metadata` (dict): Any other relevant metadata, such as image source, annotations, etc.

### Example `train.jsonl` entry:

```json
{"image_id": "circuit_001", "image_path": "electrical/circuit_diagram_001.png", "prompt": "What is the voltage across resistor R1?", "response": "The voltage across resistor R1 is 5V."}
{"image_id": "cad_assembly_005", "image_path": "mechanical/cad_assembly_005.jpg", "prompt": "Identify the fastening mechanism used for part A and part B.", "response": "Part A and Part B are fastened using M6 hex bolts."}
```

### Image Directory (`image_dir`)

This directory, specified in your configuration (e.g., `data.image_dir` in `vision_sft_config.yaml`), should contain all the image files referenced in your JSONL files.

Example structure:

```
<your_project_root>/
├── configs/
│   └── vision_sft_config.yaml
├── data/
│   ├── train.jsonl
│   ├── eval.jsonl
│   └── images/                # This is your image_dir
│       ├── electrical/
│       │   └── circuit_diagram_001.png
│       ├── mechanical/
│       │   └── cad_assembly_005.jpg
│       └── ... other images ...
└── open_vlm/
    └── ...
```

## Quantitative Reasoning Data

Data for quantitative reasoning tasks will also typically use JSONL format. The `prompt` might be more focused on extracting or calculating numerical values, and the `response` should reflect these quantitative answers accurately.

Example `quant_train.jsonl` entry:

```json
{
  "image_id": "pump_curve_002",
  "image_path": "fluid_dynamics/pump_curve_002.png",
  "prompt": "What is the flow rate in m^3/h when the head is 20m according to the pump performance curve?",
  "response": "The flow rate is approximately 15 m^3/h at a head of 20m."
}
```

## Data Augmentation

OpenVLM supports various image augmentation techniques (e.g., rotation, scaling, color jitter) configurable via the training settings. These are applied on-the-fly during training to increase dataset diversity and model robustness.

Refer to the specific configuration options (e.g., `image_augmentation`, `augmentation_strength` in `vision_sft_config.yaml`) for details.

## Creating Mock Data for Testing

For initial setup and testing, you can create small mock datasets:

**1. Create `mock_train.jsonl` and `mock_eval.jsonl` in a `data/` directory:**

   `data/mock_train.jsonl`:
   ```json
   {"image_id": "mock_img_001", "image_path": "mock_image_1.png", "prompt": "Describe this mock image.", "response": "This is a description of mock image 1."}
   {"image_id": "mock_img_002", "image_path": "mock_image_2.jpg", "prompt": "What is in mock image 2?", "response": "Mock image 2 contains a simple geometric shape."}
   ```

   `data/mock_eval.jsonl`:
   ```json
   {"image_id": "mock_img_003", "image_path": "mock_image_3.png", "prompt": "Analyze this mock image.", "response": "This is an analysis of mock image 3."}
   ```

**2. Create a `data/mock_images/` directory and add some placeholder images:**
   - `data/mock_images/mock_image_1.png`
   - `data/mock_images/mock_image_2.jpg`
   - `data/mock_images/mock_image_3.png`

   (These can be any small PNG or JPG files for testing purposes.)

**3. Update your configuration file (e.g., `vision_sft_config.yaml`) to point to these mock files:**

   ```yaml
   data:
     train_file: "./data/mock_train.jsonl"
     eval_file: "./data/mock_eval.jsonl"
     image_dir: "./data/mock_images"
   ```

This allows you to run the training pipeline and verify that data loading and basic processing are working correctly before using your actual large-scale datasets.

## Best Practices

- **Consistent Labeling:** Ensure your prompts and responses are consistently phrased and accurately reflect the image content for the task at hand.
- **Data Cleaning:** Preprocess your images and text data to remove noise or inconsistencies.
- **Balanced Datasets:** Aim for a balanced representation of different classes or scenarios if applicable to your task.
- **Sufficient Volume:** VLMs, even during fine-tuning, benefit from a substantial amount of high-quality data.

Further details on advanced data preprocessing or specific data requirements for novel modules will be added as the OpenVLM framework evolves. 