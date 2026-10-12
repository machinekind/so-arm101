# Policies

One folder per policy, named `<task>-v<n>`. Each folder holds:

- `meta.json`: the training config and evaluation results, written by the trainer.
- `README.md`: the Hugging Face link to the weights.

Weights (`*.pt`, `*.safetensors`) never go into git.

`reach-v3` and `lift-v3` arrive from the twin's `assets/policies/` with ticket #146.
Their weights move to Hugging Face at the same time.
