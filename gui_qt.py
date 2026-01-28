import os
import sys

import numpy as np
import stl.stl as stlmod
import visualizer

from PySide6.QtCore import Qt, QMimeData, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
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
    print("Интеграция VTK в Qt недоступна, STL визуализация выключена")
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


class LoginDialog(QDialog):
    """Диалоговое окно для входа в систему"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Вход в систему")
        self.setGeometry(200, 200, 400, 200)
        
        self.user_type = None  # "general" или "admin"
        self.setModal(True)  # Модальное окно
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        title_label = QLabel("Выберите тип пользователя")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Кнопка для общего пользователя
        self.general_user_button = QPushButton("Общий пользователь")
        self.general_user_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.general_user_button.clicked.connect(self._login_as_general)
        layout.addWidget(self.general_user_button)
        
        # Разделитель
        separator = QLabel("или")
        separator.setAlignment(Qt.AlignCenter)
        separator.setStyleSheet("margin: 10px; color: gray;")
        layout.addWidget(separator)
        
        # Администратор
        admin_label = QLabel("Администратор")
        admin_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(admin_label)
        
        # Поле для пароля администратора
        password_layout = QHBoxLayout()
        password_label = QLabel("Пароль:")
        password_layout.addWidget(password_label)
        
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setPlaceholderText("Введите пароль")
        password_layout.addWidget(self.password_edit)
        layout.addLayout(password_layout)
        
        # Кнопка входа как администратор
        self.admin_login_button = QPushButton("Войти как администратор")
        self.admin_login_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.admin_login_button.clicked.connect(self._login_as_admin)
        layout.addWidget(self.admin_login_button)
        
        # Метка для сообщений об ошибках
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.error_label)
        
        # Кнопка отмены
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)
        
    def _login_as_general(self):
        """Вход как общий пользователь"""
        self.user_type = "general"
        self.accept()
        
    def _login_as_admin(self):
        """Вход как администратор"""
        password = self.password_edit.text().strip()
        
        if password == "admin":
            self.user_type = "admin"
            self.accept()
        else:
            self.error_label.setText("Неверный пароль")
            self.password_edit.clear()
            
    def get_user_type(self):
        """Возвращает тип пользователя после успешного входа"""
        return self.user_type


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
        self.volume_value = "—"
        self.unit_var = "мм"

        # Сырые значения в миллиметрах (как из функций)
        self.raw_x = None
        self.raw_y = None
        self.raw_z = None
        self.raw_volume = None

        # Базовые (исходные) размеры для пропорций
        self.base_raw_x = None
        self.base_raw_y = None
        self.base_raw_z = None
        self.base_raw_volume = None

        self.stl_module = stlmod
        self.clipboard = QApplication.clipboard()

        self.visualizer = None
        self.viz_interactor = None

        self._updating_display = False  # Флаг для предотвращения рекурсии

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
            QLabel("Перетащите сюда STL файл детали")
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

        max_label = QLabel("Текущие размеры детали")
        max_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(max_label)

        dims_layout = QHBoxLayout()
        self._create_axis_block(dims_layout, "X", "длина", "x")
        self._create_axis_block(dims_layout, "Y", "ширина", "y")
        self._create_axis_block(dims_layout, "Z", "высота", "z")
        layout.addLayout(dims_layout)

        extras_layout = QHBoxLayout()
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

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Кнопка копирования размеров
        self.copy_button = QPushButton("Копировать размеры")
        self.copy_button.clicked.connect(self._copy_sizes)
        self.copy_button.setEnabled(False)
        button_layout.addWidget(self.copy_button)

        # Кнопка восстановления размеров
        self.restore_button = QPushButton("Вернуть размеры модели")
        self.restore_button.clicked.connect(self._restore_original)
        self.restore_button.setEnabled(False)
        button_layout.addWidget(self.restore_button)

        layout.addLayout(button_layout)

        layout.addStretch()

        return widget

    def _create_axis_block(self, parent_layout, axis, label, attr):
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)

        # Первая строка: ось и метка
        header_layout = QHBoxLayout()
        axis_label = QLabel(f"{axis} ({label})")
        axis_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(axis_label)
        header_layout.addStretch()
        frame_layout.addLayout(header_layout)

        # Вторая строка: поле и единица измерения
        field_layout = QHBoxLayout()
        value_edit = QLineEdit("—")
        value_edit.setStyleSheet("border: 1px solid; padding: 2px;")
        value_edit.setAlignment(Qt.AlignCenter)
        value_edit.setMinimumWidth(120)
        value_edit.setReadOnly(True)
        value_edit.setEnabled(False)
        field_layout.addWidget(value_edit)

        unit_label = QLabel("мм")
        unit_label.setFixedWidth(40)
        field_layout.addWidget(unit_label)
        field_layout.addStretch()

        frame_layout.addLayout(field_layout)

        setattr(self, f"{attr}_edit", value_edit)
        setattr(self, f"{attr}_unit_label", unit_label)

        # Подключаем сигнал returnPressed (срабатывает только при нажатии Enter)
        value_edit.returnPressed.connect(
            lambda checked=False, a=attr: self._on_dimension_changed(a)
        )

        parent_layout.addWidget(frame)

    def _create_extra_block(self, parent_layout, name, unit, attr):
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)

        # Первая строка: название
        header_layout = QHBoxLayout()
        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(name_label)
        header_layout.addStretch()
        frame_layout.addLayout(header_layout)

        # Вторая строка: поле и единица измерения
        field_layout = QHBoxLayout()
        value_label = QLabel("—")
        value_label.setStyleSheet("border: 1px solid; padding: 2px;")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setMinimumWidth(120)
        field_layout.addWidget(value_label)

        unit_label = QLabel(unit)
        unit_label.setFixedWidth(40)
        field_layout.addWidget(unit_label)
        field_layout.addStretch()

        frame_layout.addLayout(field_layout)

        setattr(self, f"{attr}_label", value_label)
        setattr(self, f"{attr}_unit_label", unit_label)

        parent_layout.addWidget(frame)

    def _pick_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите STL файл детали",
            "",
            "STL файлы (*.stl);;Все файлы (*.*)",
        )
        if file_path:
            self._handle_file(file_path)

    def _handle_file(self, file_path):
        self.selected_file = file_path
        self.file_label.setText(file_path)
        extension = os.path.splitext(file_path)[1].lower()

        if extension != ".stl":
            self.status_text = "Неподходящий файл. Нужен .stl файл"
            self.status_label.setText(self.status_text)
            self._clear_raw()
            self._update_display()
            return

        self.status_text = ""
        self.status_label.setText("")

        try:
            import time
            transform_start_time = time.time()
            result = self.stl_module.calculate_parallelepiped_volume(file_path)
            transform_end_time = time.time()
            print("---Преобразование: %s секунд ---" % (transform_end_time - transform_start_time))
            volume = result["volume"] if result else None
        except Exception as exc:
            self.status_text = f"Ошибка чтения файла: {exc}"
            self.status_label.setText(self.status_text)
            self._clear_raw()
            self._update_display()
            return

        if not result:
            self.status_text = "Не удалось вычислить габариты"
            self.status_label.setText(self.status_text)
            self._clear_raw()
            self._update_display()
            return

        # Сохраняем сырые значения в мм
        dims = result['dimensions']
        self.raw_x, self.raw_y, self.raw_z = dims
        self.raw_volume = volume

        # Сохраняем базовые размеры для пропорций
        self.base_raw_x, self.base_raw_y, self.base_raw_z = dims
        self.base_raw_volume = volume

        # Включаем поля для редактирования
        self.x_edit.setReadOnly(False)
        self.x_edit.setEnabled(True)
        self.y_edit.setReadOnly(False)
        self.y_edit.setEnabled(True)
        self.z_edit.setReadOnly(False)
        self.z_edit.setEnabled(True)
        
        # Включаем кнопку восстановления
        self.restore_button.setEnabled(True)

        # Включаем кнопку копирования
        self.copy_button.setEnabled(True)

        self._update_display()

        # Setup visualization if STL
        self._setup_visualization(extension, file_path, result)

    def _update_display(self):
        unit = self.unit_var

        # Update unit labels
        self.x_unit_label.setText(unit)
        self.y_unit_label.setText(unit)
        self.z_unit_label.setText(unit)
        self.volume_unit_label.setText(f"{unit}³")

        # Линейные размеры
        self._updating_display = True
        try:
            if self.raw_x is not None:
                conv_x = _convert_value(self.raw_x, "мм", unit)
                self.x_edit.setText(_format_dimension(conv_x, unit))
            else:
                self.x_edit.setText("—")

            if self.raw_y is not None:
                conv_y = _convert_value(self.raw_y, "мм", unit)
                self.y_edit.setText(_format_dimension(conv_y, unit))
            else:
                self.y_edit.setText("—")

            if self.raw_z is not None:
                conv_z = _convert_value(self.raw_z, "мм", unit)
                self.z_edit.setText(_format_dimension(conv_z, unit))
            else:
                self.z_edit.setText("—")
        finally:
            self._updating_display = False

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
        self.raw_volume = None

        # Отключаем поля для редактирования
        if hasattr(self, 'x_edit'):
            self.x_edit.setReadOnly(True)
            self.x_edit.setEnabled(False)
            self.y_edit.setReadOnly(True)
            self.y_edit.setEnabled(False)
            self.z_edit.setReadOnly(True)
            self.z_edit.setEnabled(False)

        # Отключаем кнопку восстановления
        if hasattr(self, 'restore_button'):
            self.restore_button.setEnabled(False)

        # Отключаем кнопку копирования
        if hasattr(self, 'copy_button'):
            self.copy_button.setEnabled(False)

    def _update_units(self, unit):
        self.unit_var = unit
        self._update_display()

    def _on_dimension_changed(self, attr):
        """Обработчик изменения одного из размеров в полях ввода"""
        if self._updating_display:
            return
            
        if self.base_raw_x is None or self.base_raw_y is None or self.base_raw_z is None:
            return
            
        # Получаем текущее значение из поля ввода
        edit_widget = getattr(self, f"{attr}_edit")
        text = edit_widget.text().strip()
        if not text or text == "—":
            return
            
        try:
            # Парсим введённое значение (уже в текущих единицах)
            entered_value = float(text.replace(',', '.'))
        except ValueError:
            return
            
        # Вычисляем базовые значения в текущих единицах (для точного сравнения)
        unit = self.unit_var
        base_x_display = _convert_value(self.base_raw_x, "мм", unit)
        base_y_display = _convert_value(self.base_raw_y, "мм", unit)
        base_z_display = _convert_value(self.base_raw_z, "мм", unit)
        
        # Вычисляем коэффициент изменения на основе отображаемых значений
        if attr == "x":
            ratio = entered_value / base_x_display
            self.raw_x = self.base_raw_x * ratio
            self.raw_y = self.base_raw_y * ratio
            self.raw_z = self.base_raw_z * ratio
        elif attr == "y":
            ratio = entered_value / base_y_display
            self.raw_x = self.base_raw_x * ratio
            self.raw_y = self.base_raw_y * ratio
            self.raw_z = self.base_raw_z * ratio
        elif attr == "z":
            ratio = entered_value / base_z_display
            self.raw_x = self.base_raw_x * ratio
            self.raw_y = self.base_raw_y * ratio
            self.raw_z = self.base_raw_z * ratio
            
        # Обновляем объём пропорционально кубу коэффициента
        if self.base_raw_volume is not None:
            self.raw_volume = self.base_raw_volume * (ratio ** 3)
            
        # Обновляем отображение
        self._update_display()

    def _restore_original(self):
        """Восстановить исходные размеры из файла"""
        if self.base_raw_x is not None:
            self.raw_x = self.base_raw_x
            self.raw_y = self.base_raw_y
            self.raw_z = self.base_raw_z
            self.raw_volume = self.base_raw_volume
            self._update_display()

    def _copy_sizes(self):
        """Копировать размеры модели в буфер обмена"""
        if self.raw_x is None or self.raw_y is None or self.raw_z is None:
            return
        unit = self.unit_var
        x_val = _convert_value(self.raw_x, "мм", unit)
        y_val = _convert_value(self.raw_y, "мм", unit)
        z_val = _convert_value(self.raw_z, "мм", unit)
        text = f"{x_val:.2f}\t{y_val:.2f}\t{z_val:.2f}"
        self.clipboard.setText(text)

    def _init_viz_widget(self, parent_layout):
        self.viz_widget = QWidget()
        parent_layout.addWidget(self.viz_widget)
        self.viz_widget.setVisible(False)

    def _setup_visualization(self, extension, file_path, result):
        if extension == ".stl" and VTK_QT_AVAILABLE:
            import time
            viz_start_time = time.time()
            if not self.visualizer:
                self.visualizer = visualizer.STLVisualizer()
            if not self.viz_interactor:
                self.viz_interactor = QVTKRenderWindowInteractor(self.viz_widget)
                layout = QVBoxLayout(self.viz_widget)
                layout.addWidget(self.viz_interactor)
                self.viz_widget.setMaximumHeight(400)
                self.visualizer.set_render_window(self.viz_interactor.GetRenderWindow())
                self.visualizer.set_interactor(self.viz_interactor.GetRenderWindow().GetInteractor())
            self.visualizer.reset_scene()
            self.visualizer.load_stl(file_path)
            self.visualizer.show_axes()
            self.visualizer.set_transform(
                result.get('rotation_matrix', None) if result else None,
                result.get('origin', None) if result else None
            )
            if result is not None:
                self.visualizer.show_bounding_box(
                    result.get("min_coords"),
                    result.get("max_coords"),
                )
            self.viz_widget.setVisible(True)
            if self.viz_interactor:
                self.viz_interactor.GetRenderWindow().Render()
            viz_end_time = time.time()
            print("---Визуализация: %s секунд ---" % (viz_end_time - viz_start_time))
        else:
            self.viz_widget.setVisible(False)



def main():
    app = QApplication(sys.argv)
    
    # Сначала показываем диалог входа
    login_dialog = LoginDialog()
    result = login_dialog.exec()
    
    # Если пользователь нажал Cancel или закрыл окно
    if result != QDialog.Accepted:
        sys.exit(0)
        
    # Получаем тип пользователя
    user_type = login_dialog.get_user_type()
    
    if user_type is None:
        sys.exit(0)
    
    # Создаем и показываем главное окно
    window = BoundingBoxApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()