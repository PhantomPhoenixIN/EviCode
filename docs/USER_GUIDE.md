# User Guide

Install dependencies as described in `SETUP.md`, then run:

```bash
python -m pytest
python scripts/run_smoke.py --config configs/smoke.yaml --output-dir experiments/smoke --resume
```

The smoke pipeline uses synthetic examples only.
