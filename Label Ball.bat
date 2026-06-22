@echo off
REM ===================================================================
REM  Double-click this file to open the Ball Labeler.
REM  No typing needed - a friendly menu opens with big buttons.
REM ===================================================================
title Ball Labeler
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "label_ball.py"
) else (
    python "label_ball.py"
)

REM If something went wrong, keep this window open so we can read the message.
if errorlevel 1 (
    echo.
    echo Something went wrong. Take a photo of this window and show Luke.
    echo.
    pause
)
