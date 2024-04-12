from __future__ import annotations

import random

from PyQt5.QtWidgets import QLabel
from qtpy.QtGui import QStandardItemModel
from qtpy.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QComboBox,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

__all__ = ["MainWindow"]

COLUMNS = [None, "Color", "Name"]


class TreeView(QTreeView):
    def __init__(self) -> None:
        super().__init__()
        self.model = QStandardItemModel()
        self.setModel(self.model)
        self.model.setHorizontalHeaderLabels(COLUMNS)

        self.button_group = QButtonGroup()

    def addGroup(self, color: None | str = None, name: str = "Class", auto_exclusive: bool = True) -> None:
        i = self.model.rowCount()

        if color:
            color_button = ColorButton(color)
            name_field = QLineEdit(name)
        else:
            random_color = self.generate_random_color_hex()
            color_button = ColorButton(random_color)
            name_field = QLineEdit(name + "_" + str(i))

        radio_button = QRadioButton("")
        self.button_group.addButton(radio_button)
        if i == 0:
            radio_button.setChecked(True)
        radio_button.setAutoExclusive(auto_exclusive)

        self.model.insertRow(i)
        radio_index = self.model.index(i, 0)
        color_index = self.model.index(i, 1)
        name_index = self.model.index(i, 2)

        self.setIndexWidget(color_index, color_button)
        self.setIndexWidget(name_index, name_field)
        self.setIndexWidget(radio_index, radio_button)

    def generate_random_color_hex(self) -> str:
        # Generate a random hex color code
        return f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._annotators: list[str] = []
        self.layout = QVBoxLayout()

        self.tree_view = TreeView()
        self.tree_view.setColumnWidth(0, 10)
        self.tree_view.setColumnWidth(1, 15)
        self.tree_view.setColumnWidth(2, 123)

        self.tree_view.setFixedWidth(150)
        self.tree_view.addGroup(color="#FFFFFF", name="undefined")

        self.layout.addWidget(self.tree_view)

        annotator_label = QLabel("Add / set annotator:")
        self.annotators = QComboBox(editable=True)
        self.annotators.lineEdit().returnPressed.connect(self._clear_focus_on_add_annotator)
        self.annotators.setToolTip(
            "Add annotator by clicking on the dropdown, typing the name of the annotator and " "pressing enter."
        )
        self.layout.addWidget(annotator_label)
        self.layout.addWidget(self.annotators)

        self._table_names: list[str] = []
        self.add_button = QPushButton("Add annotation group")
        self.add_button.clicked.connect(lambda: self.tree_view.addGroup(auto_exclusive=True))
        self.table_name_widget = QComboBox()
        self.import_button = QPushButton("Set annotation table")

        self.layout.addWidget(self.add_button)
        self.layout.addWidget(self.table_name_widget)
        self.layout.addWidget(self.import_button)

        # Set the layout on the application's window
        self.setLayout(self.layout)
        self.setWindowTitle("Annotation")
        self.show()

    def _clear_focus_on_add_annotator(self) -> None:
        self.annotators.clearFocus()


class ColorButton(QPushButton):
    def __init__(self, color: str, parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self.clicked.connect(self.openColorDialog)
        self.setStyleSheet(f"background-color: {color}")

    def openColorDialog(self) -> None:
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"background-color: {color.name()}")
