import random

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
    QTableWidget,
)

__all__ = ["MainWindow"]

COLUMNS = ["", "", "Name", "Type", "Delete"]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.index_to_color = {}
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

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["index", "class", "annotator", "color"])
        self.layout.addWidget(self.table_widget)

        self.add_annotator_widget = QLineEdit(placeholderText="Add annotator")
        self.add_annotator_widget.editingFinished.connect(self._add_annotator)
        self.annotators = QComboBox()
        self.layout.addWidget(self.add_annotator_widget)
        self.layout.addWidget(self.annotators)

        self._table_names = []
        self.add_button = QPushButton("Add annotation group")
        self.add_button.clicked.connect(lambda: self.addGroup())
        self.table_name_widget = QComboBox()
        self.import_button = QPushButton("import annotation classes from table")
        self.import_button.clicked.connect(self._import_table)
        self.export_name = QLineEdit("annotation_classes")
        self.export_button = QPushButton("export annotation classes to sdata table")

        self.layout.addWidget(self.add_button)
        self.layout.addWidget(self.table_name_widget)
        self.layout.addWidget(self.import_button)
        self.layout.addWidget(self.export_name)
        self.layout.addWidget(self.export_button)

        # Set the layout on the application's window
        self.setLayout(self.layout)
        self.setWindowTitle("Annotation")
        self.show()

    def _import_table(self):
        pass

    @property
    def table_names(self):
        return self._table_names

    @table_names.setter
    def table_names(self, table_list):
        self._table_names = table_list

    def addGroup(self, color=None, name="Class", shape="Polygon"):
        i = self.model.rowCount()

        if color:
            self.index_to_color[i] = color
            color_button = ColorButton(color)
            name_field = QLineEdit(name)
        else:
            color = self.generate_random_color_hex()
            self.index_to_color[i] = color
            color_button = ColorButton(color)
            name_field = QLineEdit(name + "_" + str(i))

        radio_button = QRadioButton("")
        self.button_group.addButton(radio_button)
        if i == 0:
            radio_button.setChecked(True)
        radio_button.setAutoExclusive(True)

        type_text = QLineEdit(shape)
        type_text.setDisabled(True)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda: self.deleteGroup())

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
            del self.index_to_color[row]
            self.model.removeRow(row)
            self.button_group.removeButton(button)

    def generate_random_color_hex(self):
        # Generate a random hex color code
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _add_annotator(self):
        annotator = self.add_annotator_widget.text()
        if annotator:
            self.annotators.addItem(annotator)


class ColorButton(QPushButton):
    def __init__(self, color, parent=None):
        super().__init__("", parent)
        self.clicked.connect(self.openColorDialog)
        self.setStyleSheet(f"background-color: {color}")

    def openColorDialog(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"background-color: {color.name()}")
