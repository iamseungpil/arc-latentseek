# ARC-LatentSeek

Integration of BARC code generation, GLM visual evaluation, and LatentSeek optimization for solving ARC (Abstraction and Reasoning Corpus) problems.

## Architecture

1. **BARC Generator**: Generates Python code to solve ARC problems
2. **Code Executor**: Executes generated code and renders outputs as images
3. **GLM Evaluator**: Uses GLM-4.1V to evaluate the quality of solutions based on visual patterns and descriptions
4. **LatentSeek Optimizer**: Optimizes BARC's hidden states based on GLM's reward signals

## Project Structure

```
arc-latentseek/
├── src/
│   ├── data/           # ARC data loading and management
│   ├── generators/     # BARC model wrapper and code generation
│   ├── executors/      # Code execution and grid rendering
│   ├── evaluators/     # GLM-based evaluation system
│   ├── optimizers/     # LatentSeek optimization
│   └── utils/          # Utilities and helpers
├── experiments/        # Experiment scripts
├── results/           # Output storage
└── tests/             # Unit tests
```

## Usage

```python
python experiments/run_single_task.py --problem_id "2a5f8217"
```

## Requirements

- PyTorch
- Transformers
- PIL (for image rendering)
- numpy
- arc (ARC dataset loader)