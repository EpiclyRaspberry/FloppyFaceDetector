# FloppyFaceDetector

something i made for a project at 3am

![mainss](./assets/mainss.png)

## Features

uses DeepFace (ArcFace model), Mediapipe tasks and OpenCV (and other shenanigans) to log your attendance using your face.

saves your attendance on an sqlite db and export it to excel

![img1](./assets/img1.png)

![img2](./assets/img2.png)

## usage

This project is sensitive to dependency versions. Use a fresh virtual environment and the pinned `requirements.txt` in this repo.

### requirements

- Python `3.11`
- a webcam
- the checked-in model file at `assets/face_landmarker.task`

### windows setup

1. Clone the repo:

```bash
git clone <repo-url>
cd FloppyFaceDetector
```

2. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Run the app:

```powershell
python main.py
```

### linux setup

1. Clone the repo:

```bash
git clone <repo-url>
cd FloppyFaceDetector
```

2. Install Tk if your distro does not already ship it.

For CachyOS / Arch with `paru`:

```bash
paru -S tk
```

For Debian / Ubuntu:

```bash
sudo apt install python3-tk
```

3. Create a virtual environment:

```bash
python3.11 -m venv .venv
```

4. Activate it.

For `fish`:

```fish
source .venv/bin/activate.fish
```

For `bash` / `zsh`:

```bash
source .venv/bin/activate
```

5. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

6. Run the app:

```bash
python main.py
```

### troubleshooting

- Do not mix this project with a global Python install. The `mediapipe` and `tensorflow` versions here are pinned to avoid protobuf conflicts.
- If you previously installed different versions into the same venv, delete the venv and recreate it instead of trying to repair it in place.
- If the camera does not open on Linux, check webcam permissions and confirm no other app is using `/dev/video0`.
