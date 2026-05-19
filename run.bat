@echo off
chcp 65001 >nul
cd /d "%~dp0"
python main.py 002050 80000 --position 100,51.44
pause
