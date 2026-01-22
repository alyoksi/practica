import importlib.util
import os
import sys

import dxf.dxf as dxfmod
import stl.stl as stlmod
import visualizer

from PySide6.QtCore import Qt, QMimeData, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_QT_AVAILABLE = True
except ImportError:
    print("VTK Qt integration not available, STL visualization disabled")
    VTK_QT_AVAILABLE = False
    QVTKRenderWindowInteractor = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



def _format_dimension(value, unit=None):
    if value is None:
        return "—"
    if unit == "мм":
        return f"{value:.2f}"
    elif unit == "см":
        return f"{value:.2f}"
    else:  # м
        if abs(value) < 0.001:
            return f"{value:.6f}"
        elif abs(value) < 0.01:
            return f"{value:.5f}"
        else:
            return f"{value:.4f}"


def _convert_value(value, from_unit, to_unit, is_square=False, is_cubic=False):
    if value is None:
        return None
    # базовые коэффициенты (из мм)
    factors = {"мм": 1.0, "см": 10.0, "м": 1000.0}
    if from_unit not in factors or to_unit not in factors:
        return value
    factor = factors[from_unit] / factors[to_unit]
    if is_square:
        factor = factor ** 2
    if is_cubic:
        factor = factor ** 3
    return value * factor


class DropArea(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("QFrame { border: 2px dashed gray; padding: 5px; }")
        self.setMinimumHeight(50)

        layout = QVBoxLayout(self)
        self.label = QLabel("Перетащите файл сюда")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.window()._handle_file(file_path)


class BoundingBoxApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Габариты детали")
        self.setGeometry(100, 100, 1100, 600)

        self.selected_file = ""
        self.status_text = ""
        self.x_value = "—"
        self.y_value = "—"
        self.z_value = "—"
        self.area_value = "—"
        self.volume_value = "—"
        self.unit_var = "мм"

        # Сырые значения в миллиметрах (как из функций)
        self.raw_x = None
        self.raw_y = None
        self.raw_z = None
        self.raw_area = None
        self.raw_volume = None

        self.dxf_module = dxfmod
        self.stl_module = stlmod

        self.visualizer = None
        self.viz_interactor = None

        self._build_ui()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_widget = self._build_left()
        right_widget = self._build_right()

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([450, 450])

    def _build_left(self):
        widget = QWidget()

        layout = QVBoxLayout(widget)

        layout.addWidget(
            QLabel("Перетащите сюда файл одной детали (dxf или stl)")
        )

        self.drop_area = DropArea(self)
        layout.addWidget(self.drop_area)

        self.pick_button = QPushButton("Выбрать файл")
        self.pick_button.clicked.connect(self._pick_file)
        layout.addWidget(self.pick_button)

        self.file_label = QLineEdit("Файл не выбран")
        self.file_label.setReadOnly(True)
        self.file_label.setStyleSheet("QLineEdit { border: none; background: transparent; }")
        layout.addWidget(self.file_label)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: red;")
        layout.addWidget(self.status_label)

        self._init_viz_widget(layout)

        layout.addStretch()

        return widget

    def _build_right(self):
        widget = QWidget()

        layout = QVBoxLayout(widget)

        layout.addWidget(
            QLabel("Максимальные размеры детали")
        )

        dims_layout = QHBoxLayout()
        self._create_axis_block(dims_layout, "X", "длина", "x")
        self._create_axis_block(dims_layout, "Y", "ширина", "y")
        self._create_axis_block(dims_layout, "Z", "высота", "z")
        layout.addLayout(dims_layout)

        # layout.addWidget(QLabel("мм"))

        extras_layout = QHBoxLayout()
        self._create_extra_block(extras_layout, "Площадь", "мм²", "area")
        self._create_extra_block(extras_layout, "Объём", "мм³", "volume")
        layout.addLayout(extras_layout)

        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("Единицы измерения:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["мм", "см", "м"])
        self.unit_combo.currentTextChanged.connect(self._update_units)
        unit_layout.addWidget(self.unit_combo)
        unit_layout.addStretch()
        layout.addLayout(unit_layout)

        layout.addStretch()

        return widget

    def _create_axis_block(self, parent_layout, axis, label, attr):
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)

        axis_label = QLabel(axis)
        axis_label.setStyleSheet("font-weight: bold;")
        frame_layout.addWidget(axis_label)

        frame_layout.addWidget(QLabel(label))

        value_label = QLabel("—")
        value_label.setStyleSheet("border: 1px solid; padding: 2px;")
        value_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(value_label)

        unit_label = QLabel("мм")
        frame_layout.addWidget(unit_label)

        setattr(self, f"{attr}_label", value_label)

        parent_layout.addWidget(frame)

    def _create_extra_block(self, parent_layout, name, unit, attr):
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)

        frame_layout.addWidget(QLabel(name))

        value_label = QLabel("—")
        value_label.setStyleSheet("border: 1px solid; padding: 2px;")
        value_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(value_label)

        frame_layout.addWidget(QLabel(unit))

        setattr(self, f"{attr}_label", value_label)

        parent_layout.addWidget(frame)

    def _pick_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл детали",
            "",
            "DXF и STL (*.dxf *.stl);;Все файлы (*.*)",
        )
        if file_path:
            self._handle_file(file_path)

    def _handle_file(self, file_path):
        self.selected_file = file_path
        self.file_label.setText(file_path)
        extension = os.path.splitext(file_path)[1].lower()

        if extension not in (".dxf", ".stl"):
            self.status_text = "Неподходящий файл. Нужен .dxf или .stl"
            self.status_label.setText(self.status_text)
            self._clear_raw()
            self._update_display()
            return

        self.status_text = ""
        self.status_label.setText("")

        try:
            if extension == ".dxf":
                dims = self.dxf_module.get_bounding_box_dimensions(file_path)
                area = self.dxf_module.bounding_rect_area(file_path)
                volume = self.dxf_module.bounding_volume(file_path)
            else:
                dims = self.stl_module.get_bounding_box_dimensions(file_path, aligned=True)
                result = self.stl_module.calculate_parallelepiped_volume(file_path)
                area = None
                volume = result["volume"] if result else None
        except Exception as exc:
            self.status_text = f"Ошибка чтения файла: {exc}"
            self.status_label.setText(self.status_text)
            self._clear_raw()
            self._update_display()
            return

        if not dims:
            self.status_text = "Не удалось вычислить габариты"
            self.status_label.setText(self.status_text)
            self._clear_raw()
            self._update_display()
            return

        # Сохраняем сырые значения в мм
        self.raw_x, self.raw_y, self.raw_z = dims
        self.raw_area = area
        self.raw_volume = volume

        # Обновляем отображение с текущими единицами
        self._update_display()

        # Setup visualization if STL
        self._setup_visualization(extension, file_path, result)

    def _update_display(self):
        unit = self.unit_var

        # Линейные размеры
        if self.raw_x is not None:
            conv_x = _convert_value(self.raw_x, "мм", unit)
            self.x_label.setText(_format_dimension(conv_x, unit))
        else:
            self.x_label.setText("—")

        if self.raw_y is not None:
            conv_y = _convert_value(self.raw_y, "мм", unit)
            self.y_label.setText(_format_dimension(conv_y, unit))
        else:
            self.y_label.setText("—")

        if self.raw_z is not None:
            conv_z = _convert_value(self.raw_z, "мм", unit)
            self.z_label.setText(_format_dimension(conv_z, unit))
        else:
            self.z_label.setText("—")

        # Площадь
        if self.raw_area is not None:
            conv_area = _convert_value(self.raw_area, "мм", unit, is_square=True)
            self.area_label.setText(_format_dimension(conv_area, unit))
        else:
            self.area_label.setText("—")

        # Объём
        if self.raw_volume is not None:
            conv_volume = _convert_value(self.raw_volume, "мм", unit, is_cubic=True)
            self.volume_label.setText(_format_dimension(conv_volume, unit))
        else:
            self.volume_label.setText("—")

    def _clear_raw(self):
        self.raw_x = None
        self.raw_y = None
        self.raw_z = None
        self.raw_area = None
        self.raw_volume = None

    def _update_units(self, unit):
        self.unit_var = unit
        self._update_display()

    def _init_viz_widget(self, parent_layout):
        self.viz_widget = QWidget()
        parent_layout.addWidget(self.viz_widget)
        self.viz_widget.setVisible(False)

    def _setup_visualization(self, extension, file_path, result):
        if extension == ".stl" and VTK_QT_AVAILABLE:
            if not self.visualizer:
                self.visualizer = visualizer.STLVisualizer()
            if not self.viz_interactor:
                self.viz_interactor = QVTKRenderWindowInteractor(self.viz_widget)
                layout = QVBoxLayout(self.viz_widget)
                layout.addWidget(self.viz_interactor)
                self.viz_widget.setMaximumHeight(400)
                self.visualizer.set_render_window(self.viz_interactor.GetRenderWindow())
                self.visualizer.set_interactor(self.viz_interactor.GetRenderWindow().GetInteractor())
            self.visualizer.load_stl(file_path)
            self.visualizer.show_axes()
            if self.raw_x is not None:
                self.visualizer.show_bounding_box(self.raw_x, self.raw_y, self.raw_z)
            self.viz_widget.setVisible(True)
            if self.viz_interactor:
                self.viz_interactor.GetRenderWindow().Render()
        else:
            self.viz_widget.setVisible(False)


def main():
    app = QApplication(sys.argv)
    window = BoundingBoxApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
