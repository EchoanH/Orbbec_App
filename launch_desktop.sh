#!/bin/bash
LOGFILE=/root/echo/app/logs/desktop_launch_$(date +%Y%m%d_%H%M%S).log
export DISPLAY=:0
export PYTHONPATH=/root/echo/qt5lib

/home/HwHiAiUser/experiment_box_host/.venv/bin/python /root/echo/app/main.py > "$LOGFILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    zenity --error --text="程序启动失败，请查看日志：$LOGFILE" --width=400 2>/dev/null || \
    xmessage "程序启动失败，请查看日志：$LOGFILE" 2>/dev/null || \
    echo "程序启动失败，日志见 $LOGFILE"
fi