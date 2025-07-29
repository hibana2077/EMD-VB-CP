@echo off
REM Windows batch script to run EMD-VB-CP experiments

echo EMD-VB-CP Tensor Completion Experiments
echo ========================================

REM Create results directory
if not exist "results" mkdir results

REM Run demo first
echo.
echo Running quick demo...
cd experiments
python demo.py
if errorlevel 1 (
    echo Demo failed
    cd ..
    pause
    exit /b 1
)

REM Ask user if they want to run full experiment
echo.
set /p choice="Demo completed. Run full experiment? (y/N): "
if /i "%choice%"=="y" (
    echo Running full experiment...
    python run_experiment.py
    if errorlevel 1 (
        echo Experiment failed
        cd ..
        pause
        exit /b 1
    )
    echo.
    echo Experiment completed! Check the results folder for outputs.
) else (
    echo Skipping full experiment.
)

cd ..
echo.
echo All done!
pause
