import random

from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

__all__ = ["MainWindow"]

COLUMNS = ["", "", "Name", "Type", "Delete"]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.tree_view = QTreeView()
        self.model = QStandardItemModel()

        self.tree_view.setModel(self.model)
        self.model.setHorizontalHeaderLabels(COLUMNS)
        self.tree_view.setColumnWidth(0, 50)
        self.tree_view.setColumnWidth(1, 50)
        self.tree_view.setColumnWidth(2, 75)
        self.tree_view.setColumnWidth(3, 75)
        self.tree_view.setColumnWidth(4, 50)

        self.tree_view.setFixedWidth(400)
        self.button_group = QButtonGroup()
        self.addGroup(color="white", name="undefined", shape="Polygon")

        self.layout.addWidget(self.tree_view)

        self.add_button = QPushButton("Add annotation group")
        self.add_button.clicked.connect(lambda: self.addGroup())

        self.layout.addWidget(self.add_button)
        # Set the layout on the application's window
        self.setLayout(self.layout)
        self.setWindowTitle("Annotation")
        self.show()

    def addGroup(self, color=None, name="Class", shape="Polygon"):
        i = self.model.rowCount()

        if color:
            color_button = ColorButton(color)
            name_field = QLineEdit(name)
        else:
            color_button = ColorButton(self.generate_random_color_hex())
            name_field = QLineEdit(name + "_" + str(i))

        radio_button = QRadioButton("")
        self.button_group.addButton(radio_button)
        if i == 0:
            radio_button.setChecked(True)
        radio_button.setAutoExclusive(True)

        type_text = QLineEdit(shape)
        type_text.setDisabled(True)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda: self.deleteGroup(model, tree_view))

        if color:
            delete_button.setDisabled(True)

        self.model.insertRow(i)
        radio_index = self.model.index(i, 0)
        color_index = self.model.index(i, 1)
        name_index = self.model.index(i, 2)
        type_index = self.model.index(i, 3)
        delete_index = self.model.index(i, 4)

        self.tree_view.setIndexWidget(color_index, color_button)
        self.tree_view.setIndexWidget(name_index, name_field)
        self.tree_view.setIndexWidget(type_index, type_text)
        self.tree_view.setIndexWidget(delete_index, delete_button)
        self.tree_view.setIndexWidget(radio_index, radio_button)

    def deleteGroup(self):
        button = self.sender()
        if button:
            row = self.tree_view.indexAt(button.pos()).row()
            self.model.removeRow(row)
            self.button_group.removeButton(button)

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
