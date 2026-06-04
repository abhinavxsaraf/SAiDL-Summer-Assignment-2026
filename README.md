### 1. Core ML

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

- Experiments were run on Google Colab using an NVIDIA T4 GPU.

- Due to compute constraints, experiments were conducted under a fixed training budget.

- Full experimental details are provided in the report.
