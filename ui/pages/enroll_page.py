"""人脸录入演示页：提供 UI 占位，不连接后端注册逻辑。"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QLineEdit, QPushButton, QVBoxLayout

from .base_page import BasePage


class EnrollPage(BasePage):
    page_title = "人脸录入"
    page_hint = "录入工作台 · 后端接口待接入"

    def __init__(self, parent=None):
        super(EnrollPage, self).__init__(parent)
        panel = QFrame()
        panel.setObjectName("formPanel")
        panel.setMaximumWidth(360)
        form = QVBoxLayout(panel)
        form.setContentsMargins(18, 18, 18, 18)
        title = QLabel("登记新成员")
        title.setObjectName("formTitle")
        form.addWidget(title)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("请输入姓名")
        form.addWidget(self.name_edit)
        self.register_button = QPushButton("注册")
        self.register_button.setObjectName("primaryButton")
        self.register_button.clicked.connect(self._register_placeholder)
        form.addWidget(self.register_button)
        self.result_label = QLabel("识别结果：等待录入\n相似度：--")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setMinimumHeight(70)
        form.addWidget(self.result_label)
        self.count_label = QLabel("已录入人员：0")
        self.count_label.setObjectName("metricValue")
        form.addWidget(self.count_label)
        self.layout().addWidget(panel, 0, Qt.AlignRight)

    def _register_placeholder(self):
        name = self.name_edit.text().strip()
        self.result_label.setText(
            "识别结果：%s\n相似度：等待后端接入" % (name or "未填写姓名"))
