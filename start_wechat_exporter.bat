@echo off
chcp 65001 >nul
echo Starting wechat-article-exporter...
echo.
docker run -d -p 3000:3000 --name wechat-exporter ghcr.io/niceday-dev/wechat-article-exporter 2>nul
if %errorlevel% neq 0 (
    echo Container may exist, trying to start existing one...
    docker start wechat-exporter 2>nul
)
echo.
echo Open http://localhost:3000 in browser
echo Login with your WeChat public account
echo.
pause
