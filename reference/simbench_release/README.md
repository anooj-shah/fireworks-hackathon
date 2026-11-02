# SimBench: A Large-Scale Benchmark for Simulating Human Behavior

[![Paper](https://img.shields.io/badge/paper-arXiv%3AXXXX.XXXXX-B31B1B.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SimBench-blue)]([https://huggingface.co/datasets/YOUR_HF_USERNAME/SimBench](https://huggingface.co/datasets/pitehu/SimBench))
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


## Overview

Simulations of human behavior using LLMs offer exciting prospects for the social and behavioral sciences. However, their utility hinges on their faithfulness to real human behaviors. SimBench addresses this by:

1.  Providing diverse datasets in a unified format (hosted on Hugging Face).
2.  Enabling measurement of various behavior types (decision-making, self-assessment, judgment, problem-solving).
3.  Covering diverse participant groups from around the world.
4.  Offering this script to generate LLM responses for robust evaluation.

Our goal with SimBench is to make progress measurable and accelerate the development of better LLM simulators.

## The SimBench Dataset

The SimBench dataset is hosted on Hugging Face: [SimBench](https://huggingface.co/datasets/pitehu/SimBench).

It contains two main splits under the `default` configuration:
*   **`SimBenchPop`**: Assesses simulation of broad population responses (7,167 test cases).
*   **`SimBenchGrouped`**: Assesses simulation of specific demographic group responses (6,343 test cases).

Each instance in the dataset includes:
*   `group_prompt_template`: Persona instruction template.
*   `group_prompt_variable_map`: Variables to fill into the persona template.
*   `input_template`: The question text.
*   `human_answer`: A dictionary of option labels and their corresponding human response proportions.
*   `dataset_name`, `group_size`, `auxiliary`: Additional metadata.

## Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch
*   Transformers
*   (Other dependencies as listed in `requirements.txt`)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/SimBench.git
    cd SimBench
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **API Keys (Optional, for API-based models):**
    The script can use API keys for OpenAI, Google, and OpenRouter. Create a JSON file named `api_keys` in the root of this repository or set environment variables:
    *   **File Method (`api_keys`):**
        ```json
        {
            "openai": "YOUR_OPENAI_API_KEY",
            "google": "YOUR_GOOGLE_API_KEY",
            "openrouter": "YOUR_OPENROUTER_API_KEY"
        }
        ```
    *   **Environment Variables:**
        ```bash
        export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
        export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
        # (Optional) export API_KEYS_PATH="/path/to/your/api_keys_file.json"
        ```
4.  **Download SimBench Data:**
    The `generate_answers.py` script expects the input data as a `.pkl` file. You can download `SimBenchPop.pkl` and `SimBenchGrouped.pkl` from the [Hugging Face Dataset page](https://huggingface.co/datasets/pitehu/SimBench/tree/main) (look under "Files and versions"). Place them in a directory accessible by the script.


## Running Simulations

The core of this repository is `generate_answers.py`. It takes a SimBench `.pkl` file as input, prompts a specified LLM, and saves the results (including LLM response distributions) to an output `.pkl` file.

### Command-Line Arguments for `generate_answers.py`:

*   `--input_file (str)`: Path to the input `.pkl` file (e.g., `SimBenchPop.pkl`).
*   `--output_file (str)`: Path to save the output `.pkl` file with LLM responses.
*   `--model_name (str)`: Name of the LLM to use.
    *   For local Hugging Face models: e.g., `mistralai/Mistral-7B-Instruct-v0.1`
    *   For OpenAI API: e.g., `gpt-3.5-turbo`, `gpt-4`
    *   For Google API: e.g., `gemini-pro`
    *   For OpenRouter API: e.g., `anthropic/claude-2` (use `--openrouter` flag)
*   `--method (str)`: Prompting method.
    *   `token_prob`: Gets the probability of the next token being one of the option labels (e.g., 'A', 'B').
    *   `verbalized`: Asks the LLM to output a JSON with estimated percentages for each option.
*   `--csd3 (str, optional)`: Environment configuration string if running on CSD3. Sets Hugging Face cache and working directory.
*   `--debug (bool, optional)`: If set, runs on a small random sample (50 instances) of the dataset for quick testing.
*   `--openrouter (bool, optional)`: If set, uses the OpenRouter API for models specified in `--model_name`.

### Example Usage:

**1. Running a local Hugging Face model (e.g., Mistral-7B-Instruct) on `SimBenchPop.pkl` using `token_prob`:**
```bash
python generate_answers.py \
    --input_file path/to/your/SimBenchPop.pkl \
    --output_file results/mistral_7b_instruct_token_prob_pop.pkl \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --method token_prob

## Evaluation

After generating LLM responses, use `calculate_simbench_score.py` to compute evaluation metrics comparing the LLM response distributions to human response distributions.


