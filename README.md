# LLM Evaluation for Chemical Engineering Q&A

This project demonstrates a framework for evaluating Large Language Models (LLMs) on chemical engineering questions and answers using LangSmith.

## Key Components

### Dataset
- Loaded from an Excel file containing chemical engineering Q&A pairs
- Converted to a Hugging Face Dataset

### Models
- Various LLMs are evaluated, including open-source models and API-based services
- A separate "judge" model (e.g., GPT-4o) is used for evaluation

### Evaluation Metrics
Custom metrics scored on a scale of 1-5 by the judge model:
- Completeness
- Relevance
- Conciseness
- Confidence
- Factuality
- Judgment

### LangSmith Integration
- Creates datasets and runs evaluations using the LangSmith platform

### Visualization
- Generates distribution plots for each evaluation metric

## Key Functions

- `get_model()`: Initializes the specified LLM
- `predict()`: Generates responses from the LLM for given questions
- `factor_evaluator()`: Evaluates LLM responses using the judge model
- `plot_figures_metrics()`: Creates visualizations of evaluation results

## Usage

1. Set up environment variables (API keys, etc.)
2. Specify the input dataset and models to evaluate
3. Run the evaluation loop, which:
   - Generates responses from each model
   - Evaluates responses using the judge model
   - Logs results to LangSmith
4. Analyze results through LangSmith dashboard and generated plots

## Notes

- The notebook is designed to work with various compute environments (local, Google Colab, RunPod, etc.)
- It includes options for CPU and GPU acceleration
- Evaluation can be resource-intensive, especially for larger models and datasets

This framework allows for systematic comparison of LLM performance on domain-specific tasks, providing insights into model capabilities and areas for improvement.