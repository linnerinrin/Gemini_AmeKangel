@echo off
rem ============================================================
rem  Gemini_AmeKangel 启动脚本
rem
rem  用法:
rem     start.bat            默认端口 5577 启动
rem     start.bat 5599       指定端口启动(原端口被占用时用这个)
rem
rem  说明:
rem     - 双击即可运行; 脚本会自动切到自身所在目录(仓库根)
rem     - 端口默认与 .env 的 PORT 一致; 若你在 .env 里改了 PORT,
rem       请把同样的端口作为参数传进来
rem     - 停止服务: 在本窗口按 Ctrl+C
rem ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=%~dp0venv\Scripts\python.exe"
set "APP_HOST=0.0.0.0"
set "APP_PORT=5577"

echo.
echo ==========================================
echo    Gemini_AmeKangel
echo ==========================================
echo.

if not exist "%PY%" (
  echo [错误] 找不到虚拟环境解释器:
  echo        %PY%
  echo.
  echo 请先创建虚拟环境并安装依赖:
  echo        python -m venv venv
  echo        venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

rem 命令行第一个参数优先作为端口
if not "%~1"=="" set "APP_PORT=%~1"

echo   监听地址 : %APP_HOST%
echo   端口     : %APP_PORT%
echo   浏览器   : http://127.0.0.1:%APP_PORT%/
echo   换端口   : start.bat 6000   ^(把 6000 换成你要的端口^)
echo.
echo   按 Ctrl+C 停止服务
echo ------------------------------------------
echo.

rem 用 uvicorn 直接按 "模块:属性" 启动: app.main 只会被导入一次,
rem 不会像 python -m app.main 那样再把模块跑第二遍。
"%PY%" -m uvicorn app.main:app --host %APP_HOST% --port %APP_PORT%
set "RC=%ERRORLEVEL%"

echo.
echo ------------------------------------------
if not "%RC%"=="0" (
  echo [服务异常退出, 退出码 %RC%]
  echo.
  echo 常见原因:
  echo   1. 端口 %APP_PORT% 已被占用。先关掉正在运行的服务, 或换端口:
  echo          start.bat 5599
  echo   2. MySQL 没启动。本应用启动时会连数据库建表与迁移。
  echo   3. 依赖缺失或配置有误, 具体看上面的报错。
  echo   4. 模型 API Key 没配。打开控制面板填好并点测试。
  echo.
) else (
  echo [服务已停止]
)
pause
endlocal
