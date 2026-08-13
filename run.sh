#!/bin/bash
set -e
export DISPLAY=:0
export PYTHONPATH=/root/echo/qt5lib
su - HwHiAiUser -c "DISPLAY=:0 xhost +local:" 2>/dev/null || true
/home/HwHiAiUser/experiment_box_host/.venv/bin/python /root/echo/app/main.py
