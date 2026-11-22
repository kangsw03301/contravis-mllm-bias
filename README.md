ContraVis: Diagnosing Language Bias and Visual Hallucinations in MLLMs
Official repository for CSE4057 (Natural Language Processing) Homework 2, Hanyang University.

1. Environment Setup
All experiments were verified on a fresh single-GPU environment (NVIDIA T4 16GB VRAM, Ubuntu 22.04, CUDA 12.1, Python 3.10).
```
git clone https://github.com/kangsw03301/contravis-mllm-bias.git
cd contravis-mllm-bias
pip install -r requirements.txt
```

2. Random Seeds
To ensure deterministic and reproducible results across model inference and nonparametric bootstrap resampling, the random seed is fixed to 42 (--seed 42) across Python random, numpy, and torch, with deterministic greedy decoding (do_sample=False, temperature=0.0).

3. Reproducing Main Results (Table 1 & Table 2)
Execute the following commands in order to reproduce the main diagnostic evaluation results (Table 1) and the attribute-level breakdown and prompt ablation (Table 2) with 95% bootstrap confidence intervals (B=2,000):
```
bash scripts/run_all_eval.sh --seed 42
python src/compute_bootstrap.py --input_dir results/ --resamples 2000 --seed 42
```

4. Repository Structure
```
contravis-mllm-bias/
├── README.md
├── requirements.txt
├── data/
│   └── contravis_meta.json
├── scripts/
│   └── run_all_eval.sh
└── src/
    ├── evaluate.py
    └── compute_bootstrap.py
```
