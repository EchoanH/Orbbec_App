#!/bin/bash
set -e
export DISPLAY=:0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/root/echo/qt5lib:${PYTHONPATH:-}
su - HwHiAiUser -c "DISPLAY=:0 xhost +local:" 2>/dev/null || true
/home/HwHiAiUser/experiment_box_host/.venv/bin/python /root/echo/app/main.py
