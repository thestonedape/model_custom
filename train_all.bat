@echo off
REM Quick Training Script for All BELT Models (Windows)
REM ====================================================

echo ============================================
echo BELT Model Training Suite
echo ============================================
echo.
echo This script trains three variants:
echo   Model 1: BELT-Ablation (no bootstrapping)
echo   Model 2: BELT-Baseline (full BELT)
echo   Model 3: BELT-Enhanced (with improvements)
echo.
echo Expected Performance:
echo   Model 1: ~25%% top-10 accuracy
echo   Model 2: ~31%% top-10 accuracy
echo   Model 3: ~37-39%% top-10 accuracy
echo ============================================
echo.

REM Check if data is prepared
if not exist ".\dataset\vocabulary.pkl" (
    echo [ERROR] Dataset not prepared!
    echo Please run: python model_custom\prepare_data.py
    exit /b 1
)

REM Ask which model to train
echo Which model would you like to train?
echo   1^) Model 1 - BELT Ablation (no bootstrapping^)
echo   2^) Model 2 - BELT Baseline (full BELT^)
echo   3^) Model 3 - BELT Enhanced (with improvements^)
echo   4^) All models (sequential^)
set /p choice="Enter choice [1-4]: "

if "%choice%"=="1" goto model1
if "%choice%"=="2" goto model2
if "%choice%"=="3" goto model3
if "%choice%"=="4" goto all_models
echo Invalid choice. Exiting.
exit /b 1

:model1
echo.
echo Training Model 1: BELT-Ablation
echo ================================
python model_custom\experiments\model_without_bootstrapping.py --config model_custom\config\belt_config.yaml --mode train
goto end

:model2
echo.
echo Training Model 2: BELT-Baseline
echo ================================
python model_custom\experiments\model_with_bootstrapping.py --config model_custom\config\belt_config.yaml --mode train
goto end

:model3
echo.
echo Training Model 3: BELT-Enhanced
echo ================================
python model_custom\experiments\model_enhanced.py --config model_custom\config\enhanced_config.yaml --mode train
goto end

:all_models
echo.
echo Training ALL models sequentially
echo ================================

echo.
echo [1/3] Training Model 1: BELT-Ablation
python model_custom\experiments\model_without_bootstrapping.py --config model_custom\config\belt_config.yaml --mode train

echo.
echo [2/3] Training Model 2: BELT-Baseline
python model_custom\experiments\model_with_bootstrapping.py --config model_custom\config\belt_config.yaml --mode train

echo.
echo [3/3] Training Model 3: BELT-Enhanced
python model_custom\experiments\model_enhanced.py --config model_custom\config\enhanced_config.yaml --mode train

echo.
echo All models trained! Results saved to:
echo   - checkpoints_ablation\
echo   - checkpoints\
echo   - checkpoints_enhanced\
goto end

:end
echo.
echo Training complete!
pause
