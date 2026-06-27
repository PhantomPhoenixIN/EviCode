# EviCode Setup

This project is designed to be reproducible across machines. Do not install dependencies silently. Every Python package must be recorded in `requirements.txt` or `requirements-dev.txt`.

## Current Machine Audit

Initial audit on 2026-06-25, followed by bootstrap updates on the same date:

- OS: Windows 10 Pro.
- CPU: AMD Ryzen 5 3600, 6 cores / 12 logical processors.
- RAM: approximately 16 GB.
- GPU: NVIDIA GeForce GTX 1660 SUPER, 6144 MiB VRAM visible through `nvidia-smi`.
- CUDA driver view: NVIDIA driver 595.97, CUDA 13.2 shown by `nvidia-smi`.
- Python: Python 3.11.9 installed via winget.
- Git: Git 2.54.0 installed via winget.
- Node.js/npm: Node.js 24.18.0 and npm 11.16.0 installed via winget. In PowerShell, use `npm.cmd` if script execution blocks the `npm.ps1` shim.
- Java/Javac: project-local Temurin JDK 17 extracted under `tools/jdk17/jdk-17.0.19+10`.
- GCC/G++: not available on PATH.
- LaTeX: MiKTeX installed via winget. `pdflatex` is available after PATH refresh.

## Required OS-Level Software

Install these manually before running project scripts:

- **Python 3.11**: required for all project scripts.
- **Git**: required for version control and future GitHub publication.
- **Node.js 20 LTS or newer**: required for JavaScript execution evidence.
- **Java JDK 17 or newer**: required for Java syntax/compile/execution evidence.
- **C/C++ compiler**: optional for later C++ support. On Windows, use MSYS2/MinGW-w64 or Visual Studio Build Tools.
- **LaTeX**: optional for paper compilation. Tectonic is recommended where available; this machine currently uses MiKTeX because Tectonic was not available through winget.

## Python Environment

From `F:\Research\EviCode`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Alternative Conda setup:

```powershell
conda env create -f environment.yml
conda activate evicode
```

## Verification

After setup:

```powershell
python --version
python -m pytest
python scripts/run_smoke.py --config configs/smoke.yaml --output-dir experiments/smoke --resume
python scripts/build_paper.py --config configs/humanevalx.yaml --output-dir paper/output --resume --force
```

The smoke test must succeed before larger datasets or experiments are attempted. Full benchmark reproduction commands are maintained in `REPRODUCIBILITY.md`.
