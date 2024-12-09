import os, sys

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, \
    QFileDialog, QLabel, QSizePolicy, QSplitter, QStackedLayout, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget
from PySide6.QtCore import Qt, QSize

class MainWindow(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Not a Default Gallery")
        self.resize(1600, 900)
        scene = QGraphicsScene(self)
        self.setScene(scene)

        # 업 레이어 구성요소 선언
        self.sideBar = sideBar()
        # 업 레이어 선언
        self.upper = QSplitter(Qt.Horizontal)
        self.upper.addWidget(sideBar())
        self.upper.addWidget(QWidget()) # 사이드바의 설정만으로 조절할 수 있으므로 즉시 객체 생성
        # 업 레이어 위젯을 그래픽 아이템으로 감싸기
        upper_proxy = QGraphicsProxyWidget()
        upper_proxy.setWidget(self.upper)

        # 다운 레이어 구성요소 선언
        blank = QWidget()

        self.imageGrid = imageGrid()
        self.imageBox = imageBox()
        # 다운 레이어의 우측 구성요소 선언: 이미지 그리드와 이미지 박스 (좌측은 사이드바와 크기가 고정된 크기가 고정된 빈 위젯)
        self.lower_right = QSplitter(Qt.Horizontal)
        self.lower_right.addWidget(self.imageGrid)
        self.lower_right.addWidget(self.imageBox)
        # 다운 레이어 및 레이아웃 선언
        self.lower_layout = QHBoxLayout()
        self.lower_layout.addWidget(blank)
        self.lower_layout.addWidget(self.lower_right)
        lower = QWidget()
        lower.setLayout(self.lower_layout)
        # 다운 레이어 위젯을 그래픽 아이템으로 감싸기
        lower_proxy = QGraphicsProxyWidget()
        lower_proxy.setWidget(lower)

        # 각 아이템 추가
        scene.addItem(upper_proxy)
        scene.addItem(lower_proxy)








# 좌측 사이드바
class sideBar(QWidget):
    def __init__(self):
        super().__init__()
        self.sideBar_layout = QVBoxLayout()

        # 레이아웃 적용
        self.setLayout(self.sideBar_layout)

# 중앙 이미지 그리드
class imageGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.imageGrid_layout = QGridLayout()

        # 레이아웃 적용
        self.setLayout(self.imageGrid_layout)

# 우측 이미지 박스
class imageBox(QWidget):
    def __init__(self):
        super().__init__()
        self.imageBox_layout = QVBoxLayout()

        # 레이아웃 적용
        self.setLayout(self.imageBox_layout)



app = QApplication(sys.argv) # event loop을 관리

window = MainWindow()
window.show() # parent가 없는 widget은 기본적으로 invisible임

app.exec()
