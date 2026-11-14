# Installation — reproducible setup

## Requirements

- Python 3.10–3.13 recommended; 3.11 preferred for scientific packages.
- On Windows, install from python.org and ensure Python is on PATH.
- Optional CUDA GPU for training; CPU is sufficient for this small physics dataset.
- Node.js 20+ **only** for the React/Three.js viewer; the built-in viewer works without Node.

## Windows PowerShell

```powershell
cd personalized-gravity-ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m gravity_ai.data.catalog_loader --validate
python -m pytest -q
python -m gravity_ai.api
```

Open http://127.0.0.1:8000/. If PowerShell prevents activation, run `.venv\Scripts\python.exe` explicitly for each Python command.

## macOS/Linux

```bash
cd personalized-gravity-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m gravity_ai.api
```

## Optional React/Three.js frontend

With the Python API still running:

```bash
cd frontend
npm install
npm run dev
```

Visit http://127.0.0.1:5173. The Vite dev server proxies `/api` calls to the Python server on port 8000.

## Docker

```bash
docker compose up --build
```

The API/demo is reachable at http://127.0.0.1:8000. This container may be large because of PyTorch.

## Why downloads were not bundled

The runtime used to build this package did not have outbound DNS access. Public raw participant measurements could not be downloaded; restricted data are not appropriate for automatic downloading. The Python code and tests can execute locally with no network. See `docs/DATASETS.md` for source-specific instructions.
