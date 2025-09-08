@echo off
echo Starting VITS Professional GUI...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    echo Please install Python or add it to PATH
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import PyQt6, qdarktheme" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements_gui.txt
    if errorlevel 1 (
        echo Error: Failed to install packages
        pause
        exit /b 1
    )
)

REM Start the GUI
echo Starting GUI...
python vits_gui.py

pause
