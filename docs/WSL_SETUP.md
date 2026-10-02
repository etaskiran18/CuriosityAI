# Running the Curiosity Organism on Windows with WSL

This guide is for a Windows laptop with an NVIDIA GPU (for example a Dell G15 with an RTX 4060).
Everything runs inside WSL (Linux on Windows), and the model runs on your GPU.

Type the commands in the **Ubuntu (WSL) terminal** unless a step says PowerShell.

## 1. Check WSL (PowerShell)

```powershell
wsl -l -v
```

You should see a distribution such as `Ubuntu` with `VERSION 2`. If WSL is not installed, run
`wsl --install` in PowerShell as administrator, restart the computer, and open "Ubuntu" from the
Start menu.

## 2. Install the tools (Ubuntu)

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip zstd
```

`zstd` is needed by the Ollama installer.

## 3. Check that WSL can see your GPU

```bash
nvidia-smi
```

You should see your GPU (for example "NVIDIA GeForce RTX 4060 Laptop GPU") and a "CUDA Version".
If the command is not found, update the NVIDIA driver **on Windows** (NVIDIA app or Dell support),
then run `wsl --shutdown` in PowerShell and open Ubuntu again. Do not install NVIDIA drivers inside WSL.

## 4. Download the project

Keep it in your Linux home folder: files there are much faster than under `/mnt/c`.

```bash
cd ~
git clone -b claude/upbeat-lovelace-bsqcwj https://github.com/etaskiran18/CuriosityAI.git
cd CuriosityAI
```

The repository is public, so no login is needed. (Once the branch is merged into `main`, you can
leave out `-b claude/upbeat-lovelace-bsqcwj`.)

## 5. Python environment

```bash
python3 --version                  # must be 3.10 or newer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m pytest                   # should end with "passed"
```

In every new terminal, first run `cd ~/CuriosityAI && source .venv/bin/activate`.

## 6. Ollama inside WSL

If Ollama for Windows is running (llama icon in the system tray), quit it first, so that two copies
of a model do not fill the 8 GB of GPU memory.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b-instruct    # about 4.4 GB
```

The installer should say "Nvidia GPU detected". If it warns that systemd is not running, Ollama
will not start by itself: open a second Ubuntu terminal and leave `ollama serve` running there.

## 7. Check the setup

```bash
python live.py --check
```

Every line should say `ok`, in particular:

* `GPU use: Ollama reports 100% of the model on the GPU`
* `GPU temperature: ...°C now`
* `Power: plugged in`

The last line says `Ready` or lists what to fix.

## 8. Let it live

```bash
python live.py --heartbeats 3      # a first short life
python live.py --status            # look inside its mind
explorer.exe memory/organism       # open its diary folder in Windows Explorer (diary.md)
python live.py --forever           # live until Ctrl+C; the mind is saved after every heartbeat
```

It rests by itself to keep the laptop cool (see "Taking care of the computer" in
[ORGANISM.md](ORGANISM.md)).

To get new versions of the code later: `cd ~/CuriosityAI && git pull`.

## If something goes wrong

| Problem | What to do |
|---|---|
| `ensurepip is not available` when creating the venv | `sudo apt install -y python3-venv` (on some versions `python3.12-venv`) |
| `Ollama: not reachable` in `--check` | run `ollama serve` in a second terminal |
| `Model ... is not installed` | `ollama pull mistral:7b-instruct` |
| `GPU use: ... CPU only` | check `nvidia-smi` (step 3), then restart: `wsl --shutdown` in PowerShell |
| `pip install` fails on `chromadb` | only v7's `run.py` needs it: `pip install requests PyYAML pydantic python-dotenv rich pytest` is enough for `live.py` |
| The laptop gets hot or loud | lower `max_gpu_temp_c` or `work_minutes` under `organism: body:` in `config.yaml` |
