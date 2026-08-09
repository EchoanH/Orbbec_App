#!/bin/bash
set -e
export DISPLAY=:0
export PYTHONPATH=/root/echo/qt5lib
/home/HwHiAiUser/experiment_box_host/.venv/bin/python /root/echo/app/main.py
