@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Голосовой ассистент — установка и запуск

rem Always work from the folder containing this BAT, regardless of the caller's current directory.
cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Не удалось перейти в папку программы.
    pause
    exit /b 1
)

echo ==========================================================
echo       ГОЛОСОВОЙ АССИСТЕНТ — УСТАНОВКА И ЗАПУСК
echo ==========================================================
echo.

rem Verify that this is the project root.
if not exist "main.py" (
    echo [ERROR] Не найден main.py.
    echo Поместите BAT-файл в корневую папку проекта рядом с main.py.
    pause
    exit /b 1
)
if not exist "requirements.txt" (
    echo [ERROR] Не найден requirements.txt.
    pause
    exit /b 1
)

rem The target system is assumed to have Python 3.13 available in PATH.
rem Use python.exe directly; do not require the optional Windows Python Launcher (py.exe).
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Команда python не найдена в PATH.
    echo Установите Python 3.13 и включите Add Python to PATH, затем откройте новое окно.
    pause
    exit /b 1
)

python --version
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] В PATH команда python запускает не Python 3.13.
    echo Ожидался Python 3.13. Проверьте команду: where python
    pause
    exit /b 1
)

rem Create a virtual environment if absent or incomplete.
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Создаю виртуальное окружение .venv...
    if exist ".venv" (
        echo [WARN] Найдено неполное .venv; удаляю только это виртуальное окружение.
        rmdir /s /q ".venv"
        if exist ".venv" (
            echo [ERROR] Не удалось удалить неполное .venv. Закройте программы, использующие его, и повторите запуск.
            pause
            exit /b 1
        )
    )
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Не удалось создать .venv.
        echo Проверьте установку Python 3.13 и наличие компонента venv.
        pause
        exit /b 1
    )
)

set "VPY=%CD%\.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo [ERROR] После создания окружения не найден %VPY%.
    pause
    exit /b 1
)

rem Install/update dependencies on first run and whenever requirements.txt changes.
set "DEPS_MARKER=.venv\requirements-installed.sha256"
set "CURRENT_HASH="
for /f "usebackq delims=" %%H in (`certutil -hashfile "requirements.txt" SHA256 2^>nul ^| findstr /r "^[0-9A-F][0-9A-F]*$"`) do if not defined CURRENT_HASH set "CURRENT_HASH=%%H"
set "CURRENT_HASH=%CURRENT_HASH: =%"
set "SAVED_HASH="
if exist "%DEPS_MARKER%" set /p SAVED_HASH=<"%DEPS_MARKER%"

if not exist "%DEPS_MARKER%" goto INSTALL_DEPS
if /i not "%CURRENT_HASH%"=="%SAVED_HASH%" goto INSTALL_DEPS

rem Sanity check: repair the environment if pip or an essential package is missing.
"%VPY%" -c "import pip, torch, transformers, faster_whisper, sounddevice, numpy, pyttsx3" >nul 2>&1
if errorlevel 1 goto INSTALL_DEPS
echo [INFO] Виртуальное окружение и зависимости уже готовы.
goto RUN_TESTS

:INSTALL_DEPS
echo.
echo [ШАГ 1/3] Обновление pip...
"%VPY%" -m ensurepip --upgrade
if errorlevel 1 (
    echo [ERROR] Не удалось подготовить pip.
    pause
    exit /b 1
)
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Не удалось обновить pip. Проверьте интернет и доступ к pypi.org.
    pause
    exit /b 1
)

echo.
echo [ШАГ 2/3] Установка зависимостей. На чистой системе это может занять время...
"%VPY%" -m pip install --prefer-binary -r "requirements.txt"
if errorlevel 1 (
    echo.
    echo [ERROR] Не удалось установить зависимости.
    echo Проверьте интернет, свободное место на диске и текст ошибки выше.
    echo После исправления причины запустите этот BAT ещё раз.
    pause
    exit /b 1
)

rem Save the marker only after a successful installation.
if not defined CURRENT_HASH (
    echo [WARN] Не удалось вычислить SHA256 requirements.txt; зависимости будут проверяться при следующем запуске.
    if exist "%DEPS_MARKER%" del /q "%DEPS_MARKER%" >nul 2>&1
) else (
    >"%DEPS_MARKER%" echo %CURRENT_HASH%
)
echo [INFO] Зависимости установлены.

:RUN_TESTS
echo.
echo [ШАГ 3/3] Проверка проекта...
if exist "tests" (
    "%VPY%" -m unittest discover -s tests -v
    if errorlevel 1 (
        echo.
        echo [WARN] Некоторые автоматические тесты завершились с ошибкой.
        echo Смотрите сообщения выше. Попытка запуска приложения не выполняется.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Папка tests не найдена; автоматические тесты пропущены.
)

echo.
echo ==========================================================
echo Запускаю ассистента.
echo При первом запуске локальные модели могут скачиваться
echo из Hugging Face и занимать несколько гигабайт.
echo Не закрывайте окно до завершения загрузки моделей.
echo ==========================================================
echo.

"%VPY%" "main.py"
set "APP_EXIT=%ERRORLEVEL%"

echo.
if not "%APP_EXIT%"=="0" (
    echo [ERROR] Приложение завершилось с кодом %APP_EXIT%.
    echo Скопируйте текст ошибки из окна для диагностики.
) else (
    echo [INFO] Ассистент завершил работу.
)
pause
exit /b %APP_EXIT%
