from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define base requirements
base_requires = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
    "peft>=0.4.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pillow>=9.5.0",
    "tqdm>=4.65.0",
    "wandb>=0.15.0", # For experiment tracking
    "hydra-core>=1.3.0", # For configuration management
    "scikit-learn>=1.2.0", # For evaluation metrics
    "matplotlib>=3.7.0", # For plotting (optional, e.g. in examples)
    "opencv-python>=4.7.0", # For image processing
    "sympy>=1.11.0", # For symbolic math in quantitative reasoning
    "ray[tune]>=2.4.0", # For hyperparameter optimization (optional)
    "diffusers>=0.18.0", # Potentially for image generation/augmentation features
    "PyYAML>=6.0" # For YAML config file parsing
]

# Define extras
extras_require = {
    "spatial": [
        "pyrender>=0.1.45", 
        "trimesh>=3.21.0"
    ],
    "diagram": [
        "pytesseract>=0.3.10", 
        "easyocr>=1.7.0"
    ],
    "dev": [
        "pytest>=7.3.1", 
        "black>=23.3.0", 
        "isort>=5.12.0",
        "pre-commit>=3.3.0", # For automated pre-commit checks
        "mypy>=1.0.0", # For static type checking
        "pylint>=2.17.0", # For linting
        "pytest-mock>=3.10.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "tox>=4.0.0",
        "build>=0.10.0"
    ],
    "datasette": [
        "datasette>=0.64.0",
        "sqlite-utils>=3.30.0"
    ],
    "llm_integration": [
        "llm>=0.9" # Assuming llm is Simon Willison's LLM tool
    ],
    "api": [
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "aiohttp>=3.8.5"
    ],
    "docs": [
        "sphinx>=6.2.1",
        "sphinx-rtd-theme>=1.2.0",
        "sphinxcontrib-mermaid>=0.9.2"
    ]
}

# Combine all extras for a 'full' installation option
extras_require["full"] = sum(extras_require.values(), [])

setup(
    name="open-vlm",
    version="0.1.0",  # Consider using dynamic version from __version__.py
    author="Jina AI",
    author_email="hello@jina.ai",
    description="OpenVLM: Pioneering Vision-Language Intelligence for Engineering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jina-ai/open-vlm",
    project_urls={
        "Documentation": "https://github.com/jina-ai/open-vlm/tree/main/docs",
        "Source Code": "https://github.com/jina-ai/open-vlm",
        "Issue Tracker": "https://github.com/jina-ai/open-vlm/issues",
        "PyPI": "https://pypi.org/project/open-vlm/",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*", "configs*"]),
    license="Apache-2.0",
    keywords=[
        "vision-language",
        "multimodal",
        "ai",
        "deep-learning",
        "computer-vision",
        "nlp"
    ],
    platforms=["any"],
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha", # Or 4 - Beta if more mature
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License", # Corrected from Apache License 2.0
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
        "Framework :: Pytest",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=base_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "open-vlm=open_vlm.cli.main:main",
        ],
        # Entry points for datasette plugins if OpenVLM provides any direct plugins
        "datasette.plugins": [
            "openvlm_datasette_plugin = open_vlm.integration.datasette_integration:plugin"
        ],
        # Entry points for llm tool plugins if OpenVLM provides any
        "llm.plugins": [
             "openvlm_llm_command = open_vlm.integration.llm_integration:register_commands"
        ]
    },
    include_package_data=True, # To include non-code files specified in MANIFEST.in
    # MANIFEST.in might be needed if you have non-Python files inside your package
    # e.g., "recursive-include open_vlm/templates *.template"
) 