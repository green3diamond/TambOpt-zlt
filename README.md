# TambOpt — TAMBO Optimization & Simulation Suite

TambOpt is a small collection of tools, models, and notebooks for simulating particle showers (via a CORSIKA-based app), generating detector responses, optimizing detector layouts, and training machine learning models to  create surrogates and optimize detector layouts. The project contains utilities for cluster job submission, detector optimization experiments, and research notebooks.

---

## Quick Highlights 
- Simulate particle showers with a C++ CORSIKA-based application (in `corsika_application/`) and submit jobs to SLURM clusters with `cluster_scripts/submit_tambo_jobs.py`.
- Optimize detector placement using an experimental class `Simulator` in `detector_optimization/`.
- Train and evaluate NN models in `ml/` using preprocessed data and utilities.
- Several notebooks to explore datasets, model architecture, and results in `notebooks/`.

---

## Structure (what’s where) 
- `corsika_application/` — C++ source for a CORSIKA-based air-shower application; build & run instructions included in `corsika_application/README.md`.
- `cluster_scripts/` — helper scripts for generating args and SBATCH scripts for cluster submission (`submit_tambo_jobs.py`).
- `detector_optimization/` — simulation and optimization tools for detector placement (has `requirements.txt` and `simulator.py`).
- `ml/` — ML training and preprocessing for NNs; see `ml/README.md` and scripts such as `gnn.py`, `nn.py`, and preprocessing utilities.
- `notebooks/` — Jupyter notebooks for EDA, training experiments, and visualization.
- `resources/` — plotting style/theme and other small resources.
- `data/` — sample data (e.g., `first_10k_rows.csv`) used for initial detector optimization script.

---

## Quick Start (local development) 
1. Clone repository:

```powershell
git clone https://github.com/TAMBO-Observatory/TambOpt.git
cd TambOpt
```

2. Python dependencies (recommended in an isolated env):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r detector_optimization/requirements.txt
# ML pipeline requires PyTorch & PyTorch Geometric
```

3. Run CORSIKA/C++ using bash scripts  (see `corsika_application/README.md`):

4. Generate cluster jobs (slurm) for multiple simultions (`cluster_scripts`):

5. Train surrogate model

6. Run detector optimization scripts

---

## Key Usage Notes & Tips 💡
- CORSIKA and the C++ simulation require third-party dependencies, and the build assumes a FLUKA-enabled environment and Conan dependencies; follow the `corsika_application/README.md` file for cluster-specific details and env var settings.
- `cluster_scripts/submit_tambo_jobs.py` generates argument files and a reusable SBATCH template.
- The ML code uses PyTorch.
- Example dataset and quick debugging/test usage: `data/first_10k_rows.csv` is a small CSV example used for detector optimization tests.

---

## Notebooks & Examples 🧪
- `notebooks/` houses exploratory notebooks for: dataset EDA, model training checkpoints, architecture visualization, and simulator experiments (e.g., `simulator_test_tambo.ipynb`, `tambo_data_exploration.ipynb`).
