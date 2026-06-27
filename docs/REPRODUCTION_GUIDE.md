# Reproduction Guide

The current reproducible target is the synthetic smoke test:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest
python scripts/run_smoke.py --config configs/smoke.yaml --output-dir experiments/smoke --resume
python scripts/build_paper.py --config configs/smoke.yaml --output-dir paper/output --resume
```

HumanEval-X reproduction instructions will be added after the downloader and preprocessing pipeline are implemented.
