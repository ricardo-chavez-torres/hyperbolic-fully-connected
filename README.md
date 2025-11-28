## Install dependencies

1. Clone the repo:
```bash
git clone https://github.com/ricardo-chavez-torres/hyperbolic-fully-connected.git
cd hyperbolic-fully-connected
```
2. Install `uv` if necessary

Linux and macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Or go to [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)


3. Install dependencies with `uv`
```bash
uv venv
uv sync
```