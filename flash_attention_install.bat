@echo off

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo This installation process can take several hours to complete.
echo Installing dependencies...
pip install ninja
echo Installing flash-attention...
uv pip install flash-attn --no-build-isolation
echo Installing Triton from source...
git clone https://github.com/triton-lang/triton.git
cd triton
pip install cmake wheel
pip install -e .
echo Flash-attention and Triton installation completed.
pause 