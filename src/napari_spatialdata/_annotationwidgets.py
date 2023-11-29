from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTreeView,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QComboBox,
    QColorDialog,
    QRadioButton,
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import QSize, Qt
import sys
import random

__all__ = ["MainWindow"]

COLUMNS = ["", "", "Name", "Type", "Delete"]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        tree_view = QTreeView()
        model = QStandardItemModel()

        tree_view.setModel(model)
        model.setHorizontalHeaderLabels(COLUMNS)
        tree_view.setColumnWidth(0, 50)
        tree_view.setColumnWidth(1, 50)
        tree_view.setColumnWidth(2, 75)
        tree_view.setColumnWidth(3, 75)
        tree_view.setColumnWidth(4, 50)

        tree_view.setFixedWidth(400)
        self.addGroup(model, tree_view, color="white", name="undefined", shape="")

        layout.addWidget(tree_view)

        add_button = QPushButton("Add layer")
        add_button.clicked.connect(lambda: self.addGroup(model, tree_view))

        layout.addWidget(add_button)
        # Set the layout on the application's window
        self.setLayout(layout)
        self.setWindowTitle("Annotation")
        self.show()

    def addGroup(self, model, tree_view, color=None, name="Name", shape="Polygon"):
        i = model.rowCount()

        if color:
            color_button = ColorButton(color)
            name_field = QLineEdit(name)
        else:
            color_button = ColorButton(self.generate_random_color_hex())
            name_field = QLineEdit(name + "_" + str(i))

        radio_button = QRadioButton("")
        if i == 0:
            radio_button.setChecked(True)
        radio_button.setAutoExclusive(True)

        type_text = QLineEdit(shape)
        type_text.setDisabled(True)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda: self.deleteGroup(model, tree_view))

        if color:
            delete_button.setDisabled(True)

        model.insertRow(i)
        radio_index = model.index(i, 0)
        color_index = model.index(i, 1)
        name_index = model.index(i, 2)
        type_index = model.index(i, 3)
        delete_index = model.index(i, 4)

        tree_view.setIndexWidget(color_index, color_button)
        tree_view.setIndexWidget(name_index, name_field)
        tree_view.setIndexWidget(type_index, type_text)
        tree_view.setIndexWidget(delete_index, delete_button)
        tree_view.setIndexWidget(radio_index, radio_button)

    def deleteGroup(self, model, tree_view):
        button = self.sender()
        if button:
            row = tree_view.indexAt(button.pos()).row()
            model.removeRow(row)

    def generate_random_color_hex(self):
        # Generate a random hex color code
        color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return color


class ColorButton(QPushButton):
    def __init__(self, color, parent=None):
        super().__init__("", parent)
        self.clicked.connect(self.openColorDialog)
        self.setStyleSheet(f"background-color: {color}")

    def openColorDialog(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"background-color: {color.name()}")
