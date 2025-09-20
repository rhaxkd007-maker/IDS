\
    @echo off
    REM === User path to dataset ===
    set DATA_DIR=C:\Users\Administrator\Desktop\ids\dataset
    set OUT_DIR=%CD%\out
    if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
    echo Using DATA_DIR=%DATA_DIR%
    echo OUT_DIR=%OUT_DIR%
    python -m src.data_prep --data_dir "%DATA_DIR%" --out_dir "%OUT_DIR%\data"
    if errorlevel 1 goto :eof
    python -m src.train_mlp --data_dir "%OUT_DIR%\data" --out_dir "%OUT_DIR%" --epochs 40 --batch_size 512
    if errorlevel 1 goto :eof
    python -m src.evaluate --data_dir "%OUT_DIR%\data" --out_dir "%OUT_DIR%"
