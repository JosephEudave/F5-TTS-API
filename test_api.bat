@echo off
echo Testing F5-TTS API...

echo Activating virtual environment...
call .venv\Scripts\activate.bat

set /p input_text="Enter text to generate speech: "

echo Running API tests...
python test_api.py "%input_text%"
pause 