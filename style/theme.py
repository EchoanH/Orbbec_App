"""集中定义深色工业风 QSS，避免依赖 Qt 默认控件外观。"""

APP_STYLE = """
QWidget {
    color: #e7edf5;
    background: #171b25;
    font-family: "Droid Sans Fallback";
}
QMainWindow, QStackedWidget { background: #171b25; }
QFrame#sidebar { background: #11151e; border-right: 1px solid #2a3140; }
QLabel#brand { color: #f7fafc; font-size: 22px; font-weight: 700; }
QLabel#brandSub { color: #7d899c; font-size: 11px; }
QLabel#sectionLabel { color: #66758b; font-size: 11px; font-weight: 700; }
QPushButton#navButton {
    color: #93a1b5; background: transparent; border: 0; border-left: 3px solid transparent;
    text-align: left; padding: 14px 16px; font-size: 14px;
}
QPushButton#navButton:hover { color: #eef5ff; background: #1d2431; }
QPushButton#navButton:checked { color: #f7fbff; background: #222c3a; border-left-color: #4a9eff; }
QPushButton#closeButton { color: transparent; background: transparent; border: 0; }
QFrame#topBar { background: #1b202c; border-bottom: 1px solid #2a3140; }
QLabel#pageTitle { color: #f7fafc; font-size: 20px; font-weight: 700; }
QLabel#pageHint { color: #7d899c; font-size: 12px; }
QLabel#statusPill { color: #9fe8cf; background: #19392f; border: 1px solid #28674f; border-radius: 4px; padding: 6px 10px; }
QLabel#fpsLabel { color: #a6b3c4; font-size: 12px; padding-right: 10px; }
QFrame#content { background: #171b25; }
QFrame#videoPanel { background: #0d1118; border: 1px solid #2d3645; border-radius: 6px; }
QLabel#videoLabel { background: #0b0e14; color: #65738a; border-radius: 5px; }
QLabel#placeholderTitle { color: #c8d2df; font-size: 22px; font-weight: 700; }
QLabel#placeholderText { color: #748196; font-size: 13px; }
QFrame#resultBar { background: #1d2430; border: 1px solid #2b3545; border-radius: 5px; }
QLabel#resultLabel { color: #d4dde8; font-size: 13px; }
QLabel#metricValue { color: #f2f7fc; font-size: 18px; font-weight: 700; }
QLabel#metricCaption { color: #748196; font-size: 11px; }
QFrame#formPanel { background: #1d2430; border: 1px solid #2b3545; border-radius: 6px; }
QLabel#formTitle { color: #f2f7fc; font-size: 16px; font-weight: 700; }
QLineEdit { color: #f2f7fc; background: #11161f; border: 1px solid #364255; border-radius: 4px; padding: 10px; }
QLineEdit:focus { border-color: #4a9eff; }
QPushButton#primaryButton { color: #08131f; background: #4a9eff; border: 0; border-radius: 4px; padding: 10px 18px; font-weight: 700; }
QPushButton#primaryButton:hover { background: #6bb2ff; }
QPushButton#primaryButton:pressed { background: #347fda; }
QScrollBar:vertical { width: 8px; background: #171b25; }
QScrollBar::handle:vertical { background: #364255; border-radius: 4px; }
"""
