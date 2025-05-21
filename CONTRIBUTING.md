# Contributing to OpenVLM

First off, thank you for considering contributing to OpenVLM! We welcome contributions from everyone, and we're excited to see how you can help make this project even better.

This document provides guidelines for contributing to OpenVLM. Please read it carefully to ensure a smooth and effective contribution process.

## How Can I Contribute?

There are many ways to contribute to OpenVLM, including but not limited to:

*   **Reporting Bugs:** If you find a bug, please open an issue on our [GitHub Issue Tracker](https://github.com/jina-ai/open-vlm/issues), providing detailed information about the bug and how to reproduce it.
*   **Suggesting Enhancements:** Have an idea for a new feature or an improvement to an existing one? Open an issue to discuss it.
*   **Writing Code:** Implement new features, fix bugs, or improve existing code.
*   **Improving Documentation:** Help us make our documentation clearer, more comprehensive, and more accurate.
*   **Adding Examples:** Create new examples or improve existing ones to showcase OpenVLM's capabilities.
*   **Submitting Pull Requests:** For code, documentation, or example changes.
*   **Participating in Discussions:** Share your thoughts and ideas on GitHub Discussions or in our community channels (when available).
*   **Testing:** Help us test new releases and features.

## Getting Started

1.  **Fork the Repository:** Click the "Fork" button on the [OpenVLM GitHub page](https://github.com/jina-ai/open-vlm) to create your own copy of the repository.
2.  **Clone Your Fork:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/open-vlm.git
    cd open-vlm
    ```
3.  **Set Up Your Development Environment:**
    *   We recommend using a virtual environment (e.g., Conda or `venv`):
        ```bash
        # Using Conda
        conda create -n openvlm-dev python=3.9 -y
        conda activate openvlm-dev
        
        # Or using venv
        # python -m venv .venv
        # source .venv/bin/activate # On Linux/macOS
        # .venv\Scripts\activate    # On Windows
        ```
    *   Install OpenVLM in editable mode with development dependencies:
        ```bash
        pip install -e ".[dev,spatial,diagram]"
        ```
        This ensures that your changes to the source code are immediately reflected when you run the package, and installs tools like `pytest`, `black`, and `isort`.
4.  **Install Pre-commit Hooks (Recommended):**
    OpenVLM uses pre-commit hooks to automatically format code and run linters before each commit. This helps maintain code quality and consistency.
    ```bash
    pre-commit install
    ```
    Now, `black` and `isort` (and potentially other checks) will run automatically when you `git commit`.

## Making Changes

1.  **Create a New Branch:** Create a descriptive branch for your changes:
    ```bash
    git checkout -b feature/your-awesome-feature # For new features
    # or
    git checkout -b fix/issue-123-bug-fix       # For bug fixes
    ```
2.  **Write Your Code:** Make your changes, write new code, or update documentation.
    *   Follow our coding style (see below).
    *   Write clear and concise commit messages.
    *   Add tests for any new functionality or bug fixes.
3.  **Test Your Changes:**
    Run the test suite to ensure your changes haven't introduced regressions:
    ```bash
    pytest tests/
    ```
    If you added new features, ensure you've also added corresponding tests.
4.  **Format and Lint:**
    Even if you have pre-commit hooks, you can run them manually:
    ```bash
    pre-commit run --all-files
    # or to run black and isort manually:
    # black .
    # isort .
    ```
5.  **Commit Your Changes:**
    ```bash
    git add .
    git commit -m "feat: Implement awesome feature X"
    # or
    # git commit -m "fix: Resolve issue #123 by doing Y"
    ```
    (See our Commit Message Guidelines below)

## Submitting a Pull Request (PR)

1.  **Push Your Changes:** Push your branch to your fork on GitHub:
    ```bash
    git push origin feature/your-awesome-feature
    ```
2.  **Open a Pull Request:** Go to the OpenVLM GitHub repository and click the "New pull request" button. Choose your fork and branch to compare with the `main` branch of the `jina-ai/open-vlm` repository.
3.  **Describe Your PR:** Provide a clear and concise description of your changes. Explain the problem you're solving or the feature you're adding. If your PR addresses an existing issue, link to it (e.g., "Closes #123").
4.  **Code Review:** At least one core contributor will review your PR. Be prepared to address feedback and make changes if requested.
5.  **Merging:** Once your PR is approved and passes all CI checks, it will be merged into the `main` branch.

## Coding Style and Conventions

*   **Python Code:** Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.
*   **Formatting:** We use `black` for code formatting and `isort` for import sorting. Pre-commit hooks should handle this automatically.
*   **Type Hinting:** Use Python type hints for all function signatures and important variables. We aim for full type coverage.
*   **Docstrings:** Write clear and informative docstrings for all public modules, classes, and functions. We generally follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings (similar to NumPy/SciPy style).
*   **Testing:** Write tests using `pytest`. Ensure good test coverage for your contributions.
*   **Imports:** Organize imports as follows: standard library, third-party libraries, then first-party (OpenVLM) imports, each group separated by a blank line. `isort` handles this.

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This makes it easier to understand the project's history and automate changelog generation.

A commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Common Types:**

*   `feat`: A new feature.
*   `fix`: A bug fix.
*   `docs`: Documentation only changes.
*   `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).
*   `refactor`: A code change that neither fixes a bug nor adds a feature.
*   `perf`: A code change that improves performance.
*   `test`: Adding missing tests or correcting existing tests.
*   `build`: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm).
*   `ci`: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs).
*   `chore`: Other changes that don't modify `src` or `test` files.

**Example Commit Messages:**

```
feat: Add support for QLoRA fine-tuning
fix: Correct image path resolution in SFT dataset
docs: Update installation guide with troubleshooting tips
refactor(cli): Improve argument parsing in train command
```

## Development Environment Setup Tips

*   **IDE:** Use an IDE that supports Python, `black`, `isort`, and `pytest` (e.g., VS Code with Python extensions).
*   **Stay Updated:** Regularly pull changes from the upstream `main` branch to keep your local fork up-to-date:
    ```bash
    git remote add upstream https://github.com/jina-ai/open-vlm.git # Do this once
    git fetch upstream
    git rebase upstream/main
    ```

## Code of Conduct

OpenVLM follows the [Jina AI Code of Conduct](https://github.com/jina-ai/jina/blob/master/CODE_OF_CONDUCT.md). Please ensure you read and adhere to it.

## Questions?

If you have any questions, feel free to open an issue or ask on our community channels.

Thank you for contributing to OpenVLM! We appreciate your support. 