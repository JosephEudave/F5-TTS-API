@echo off
echo Setting up F5-TTS with UV...

echo Creating UV environment...
uv venv .venv
call .venv\Scripts\activate.bat

echo Installing UV...
pip install uv

echo Installing PyTorch with CUDA support...
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo Installing required packages...
uv pip install pyyaml safetensors huggingface_hub fastapi uvicorn ninja
uv pip install transformers datasets accelerate
uv pip install websockets

echo Installing ffmpeg...
uv pip install ffmpeg-python

echo Installing F5-TTS in editable mode...
uv pip install -e .

echo Creating models directory structure...
if not exist "models\vocos" mkdir "models\vocos"

echo Downloading Vocos model...
python -c "from huggingface_hub import snapshot_download; snapshot_download('charactr/vocos-mel-24khz', local_dir='models/vocos')"

echo Installation complete! You can now activate the environment using: .venv\Scripts\activate.bat
pause 