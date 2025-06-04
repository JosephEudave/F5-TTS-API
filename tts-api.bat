@echo off
echo Starting F5-TTS API server...

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting the API server...
uvicorn app:app --host 0.0.0.0 --port 9000
if errorlevel 1 (
    echo Error occurred while running the application
    echo Please check the error message above
    pause
    exit /b 1
)
pause 