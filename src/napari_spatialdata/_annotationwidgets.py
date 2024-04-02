import random

from qtpy.QtGui import QStandardItemModel
from qtpy.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QComboBox,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableView,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

__all__ = ["MainWindow"]

COLUMNS = [None, "Color", "Name"]


class TreeView(QTreeView):
    def __init__(self):
        super().__init__()
        self.model = QStandardItemModel()
        self.setModel(self.model)
        self.model.setHorizontalHeaderLabels(COLUMNS)

        self.button_group = QButtonGroup()

    def addGroup(self, color=None, name="Class", auto_exclusive=True):
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

    def generate_random_color_hex(self):
        # Generate a random hex color code
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.tree_view = TreeView()
        # self.model = QStandardItemModel()
        #
        # self.tree_view.setModel(self.model)
        # self.model.setHorizontalHeaderLabels(COLUMNS)
        self.tree_view.setColumnWidth(0, 10)
        self.tree_view.setColumnWidth(1, 15)
        self.tree_view.setColumnWidth(2, 50)

        self.tree_view.setFixedWidth(120)
        self.tree_view.addGroup(color="#FFFFFF", name="undefined")

        self.layout.addWidget(self.tree_view)

        # We use a tableview instead of table widget as the setModel tablewidget method is private.
        self.table_widget = QTableView()  # QTableWidget()
        self.layout.addWidget(self.table_widget)

        self.add_annotator_widget = QLineEdit(placeholderText="Add annotator")
        self.add_annotator_widget.editingFinished.connect(self._add_annotator)
        self.annotators = QComboBox()
        self.layout.addWidget(self.add_annotator_widget)
        self.layout.addWidget(self.annotators)

        self._table_names = []
        self.add_button = QPushButton("Add annotation group")
        self.add_button.clicked.connect(lambda: self.tree_view.addGroup(auto_exclusive=True))
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
