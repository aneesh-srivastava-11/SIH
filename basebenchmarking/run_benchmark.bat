@echo off
echo ==========================================
echo Image Registration Benchmark Setup ^& Run
echo ==========================================

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Running benchmark pipeline...
python -m pipeline.runner

echo.
echo Starting dashboard (accessible at http://localhost:5001)...
python -m dashboard.app

pause
