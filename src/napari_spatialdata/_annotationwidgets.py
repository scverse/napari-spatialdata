from __future__ import annotations

import random

from qtpy.QtCore import QModelIndex, Signal
from qtpy.QtGui import QStandardItemModel
from qtpy.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QComboBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

__all__ = ["MainWindow"]

COLUMNS = [None, "color", "class"]


class TreeView(QTreeView):
    color_button_added = Signal(QPushButton)
    class_name_text_added = Signal(QLineEdit)

    def __init__(self) -> None:
        super().__init__()
        self.model = QStandardItemModel()
        self.setModel(self.model)
        self.model.setHorizontalHeaderLabels(COLUMNS)

        self.header().setSectionsClickable(True)
        self.button_group = QButtonGroup()

        # Required as on mac the selected indexes never get updated.
        self.button_to_color_index: dict[QPushButton, QModelIndex] = {}
        self.button_to_class_index: dict[QPushButton, QModelIndex] = {}
        self.color_button_to_class_line_edit: dict[QPushButton, QLineEdit] = {}

    def addGroup(self, color: None | str = None, name: str = "Class", auto_exclusive: bool = True) -> None:
        i = self.model.rowCount()

        if color:
            color_button = ColorButton(color, i)
            name_field = QLineEdit(name)
            if i == 0:
                name_field.setReadOnly(True)
                name_field.setToolTip(
                    "The first row of the table is not editable. " "Added annotation groups will be editable."
                )
        else:
            random_color = self.generate_random_color_hex()
            color_button = ColorButton(random_color, i)
            name_field = QLineEdit(name + "_" + str(i))
            name_field.returnPressed.connect(lambda: self.clear_name_field_focus(name_field))

        radio_button = QRadioButton("")
        self.button_group.addButton(radio_button)
        if i == 0:
            radio_button.setChecked(True)
        radio_button.setAutoExclusive(auto_exclusive)

        self.model.insertRow(i)
        radio_index = self.model.index(i, 0)
        color_index = self.model.index(i, 1)
        name_index = self.model.index(i, 2)

        self.button_to_color_index[radio_button] = color_index
        self.button_to_class_index[radio_button] = name_index
        self.color_button_to_class_line_edit[color_button] = name_field
        self.setIndexWidget(color_index, color_button)
        self.setIndexWidget(name_index, name_field)
        self.setIndexWidget(radio_index, radio_button)

        self.color_button_added.emit(color_button)
        self.class_name_text_added.emit(name_field)

    def generate_random_color_hex(self) -> str:
        # Generate a random hex color code
        return f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"

    def reset_to_default_tree_view(self) -> None:
        if self.model.rowCount() != 1:
            count = self.model.rowCount()
            self.model.removeRows(1, count - 1)
            self.reset_button_group()
            self.button_group.buttons()[0].setChecked(True)

    def set_class_column_header(self, class_col_name: str) -> None:
        columns_to_set = COLUMNS.copy()
        columns_to_set[-1] = class_col_name
        self.model.setHorizontalHeaderLabels(columns_to_set)

    def reset_class_column_header(self) -> None:
        self.model.setHorizontalHeaderLabels(COLUMNS)

    def reset_button_group(self) -> None:
        for button in self.button_group.buttons()[1:]:
            self.button_group.removeButton(button)

    def clear_name_field_focus(self, name_field: QLineEdit) -> None:
        name_field.clearFocus()
        # for some reasons, in some macs the focus is not removed with the line above
        self.setFocus()


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._annotators: list[str] = []
        self.layout = QVBoxLayout()

        self.link_button = QPushButton("Link layer to sdata")
        self.link_button.setToolTip(
            "Link layer to sdata object present in viewer if no sdata object is associated with"
            " layer. Only 1 active spatialdata object must be present in viewer or in "
            "selection. Press Shift+L for a shortcut."
        )
        self.layout.addWidget(self.link_button)

        self.tree_view = TreeView()
        self.tree_view.addGroup(color="#FFFFFF", name="undefined")

        self.tree_view.header().setSectionResizeMode(0, QHeaderView.Fixed)
        self.tree_view.header().resizeSection(0, 40)
        self.tree_view.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.add_button = QPushButton("Add annotation group")
        self.add_button.clicked.connect(lambda: self.tree_view.addGroup(auto_exclusive=True))
        self.layout.addWidget(self.add_button)
        self.layout.addWidget(self.tree_view)

        self.description_box = QTextEdit()
        self.description_box.setPlaceholderText(
            "Add your description for selected element here and press button below."
        )
        self.description_box.setMaximumHeight(self.description_box.fontInfo().pixelSize() * 8)
        self.layout.addWidget(self.description_box)

        annotator_label = QLabel("Add / set annotator:")
        self.annotators = QComboBox(editable=True)
        self.annotators.lineEdit().returnPressed.connect(self._clear_focus_on_add_annotator)
        self.annotators.setToolTip(
            "Add annotator by clicking on the dropdown, typing the name of the annotator and pressing enter."
        )
        self.layout.addWidget(annotator_label)
        self.layout.addWidget(self.annotators)

        self.set_annotation = QPushButton("Set description, class and annotator")
        self.layout.addWidget(self.set_annotation)

        table_label = QLabel("Annotation table:")
        self._table_names: list[str] = []
        self.table_name_widget = QComboBox()

        self.layout.addWidget(table_label)
        self.layout.addWidget(self.table_name_widget)

        self.save_button = QPushButton("Save selected annotation layer")
        self.save_button.setToolTip(
            "Save annotation as Shapes element with associated Table. Requires current active "
            "layer to be a Shapes layer that is linked to a SpatialData object. Shortcut is "
            "pressing shift + E."
        )
        self.layout.addWidget(self.save_button)

        # Set the layout on the application's window
        self.setLayout(self.layout)
        self.setWindowTitle("Annotation")
        self.show()

    def _clear_focus_on_add_annotator(self) -> None:
        self.annotators.clearFocus()


class ColorButton(QPushButton):
    color_changed = Signal(str, QPushButton)

    def __init__(self, color: str, button_index: int, parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        if button_index != 0:
            self.clicked.connect(self.openColorDialog)
        self.setStyleSheet(f"background-color: {color}")

    def openColorDialog(self) -> None:
        color = QColorDialog.getColor(parent=self.parent())
        if color.isValid():
            self.setStyleSheet(f"background-color: {color.name()}")
            self.color_changed.emit(color.name(), self)
