@echo off
echo Running F5-TTS CLI...

echo Activating virtual environment...
call .venv\Scripts\activate.bat

set /p input_text="Enter text to generate speech: "

echo Running TTS CLI...
python -m f5_tts.infer.infer_cli --config custom.toml --gen_text "%input_text%"

pause 