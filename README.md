## 1. Core ML

Implemented and evaluated architectural approaches for long-context modeling in decoder-only Transformers using WikiText-2.

Components explored include:

- Transformer baseline

- Attention variants:
  - Grouped Query Attention (GQA)
  - Sliding Window Attention
  - Linear Attention

- Positional encodings:
  - RoPE
  - ALiBi
  - Relative Positional Encoding

- Convolution-attention hybrids:
  - Conv-Augmented Attention
  - Gated Convolutional FFN

- Final combined model using best-performing components.

## Files

- `Transformers_Long_Context_Modeling.pdf`  
  Full report containing methodology, experiments, results, and analysis.

- `Transformers_long_context_modeling.ipynb`  
  Google Colab notebook containing implementation and experiments.

## Running the Code

The implementation was developed and run in Google Colab.

Open:

```bash
Transformers_long_context_modeling.ipynb
```

and run in Colab or Jupyter.

## 2. Sparsity and Optimization

Implemented and evaluated parameter-efficient fine-tuning (PEFT) methods and sparse optimization techniques on the CoLA benchmark using DeBERTa-v3-base.

Components explored include:

### PEFT Methods
- LoRA (Low-Rank Adaptation)
- AdaLoRA (Adaptive Budget Allocation)
- SoRA (Sparse Low-Rank Adaptation)

### Sparse Optimization
- SGD with ℓ1 subgradients
- Proximal Gradient Descent

## Files

### Sparsity_and_Optimization.pdf
Full report containing theoretical background, derivations, experiments, results, and analysis.

### Sparsity_and_Optimization.ipynb
Google Colab notebook containing implementation and experiments.

## Running the Code

The implementation was developed and run in Google Colab.

Open:

`Sparsity_and_Optimization.ipynb`

and run in Colab or Jupyter.

Experiments were conducted using:

- DeBERTa-v3-base (184M parameters)
- CoLA dataset from the GLUE benchmark

Full experimental details and results are provided in the report.
