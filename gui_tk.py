import importlib.util
import os
import tkinter as tk
from tkinter import filedialog, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # pragma: no cover - опциональная зависимость
    DND_FILES = None
    TkinterDnD = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(module_name, relative_path):
    module_path = os.path.join(BASE_DIR, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _extract_first_path(data, root):
    if not data:
        return ""
    if hasattr(root, "tk"):
        paths = root.tk.splitlist(data)
    else:
        paths = data.split()
    return paths[0] if paths else ""


class BoundingBoxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Габариты детали")
        self.root.geometry("900x500")

        self.selected_file = tk.StringVar(value="Файл не выбран")
        self.status_text = tk.StringVar(value="")
        self.x_value = tk.StringVar(value="—")
        self.y_value = tk.StringVar(value="—")
        self.z_value = tk.StringVar(value="—")
        self.area_value = tk.StringVar(value="—")
        self.volume_value = tk.StringVar(value="—")
        self.unit_var = tk.StringVar(value="мм")

        # Сырые значения в миллиметрах (как из функций)
        self.raw_x = None
        self.raw_y = None
        self.raw_z = None
        self.raw_area = None
        self.raw_volume = None

        # Метки единиц измерения для динамического обновления
        self.unit_labels = {}

        self.dxf_module = _load_module("dxf_module", os.path.join("dxf", "dxf.py"))
        self.stl_module = _load_module("stl_module", os.path.join("stl", "stl.py"))

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        right_frame = ttk.Frame(main_frame)

        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(20, 0))

        separator = ttk.Separator(main_frame, orient="vertical")
        separator.grid(row=0, column=1, sticky="ns")

        self._build_left(left_frame)
        self._build_right(right_frame)

    def _build_left(self, parent):
        ttk.Label(
            parent,
            text="Перетащите сюда файл одной детали (dxf или stl)",
            font=("Segoe UI", 11, "bold"),
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        drop_frame = ttk.Frame(parent)
        drop_frame.pack(fill=tk.X)

        self.drop_area = tk.Label(
            drop_frame,
            text="Перетащите файл сюда",
            relief="ridge",
            borderwidth=2,
            height=5,
            justify="center",
            background="white",
        )
        self.drop_area.pack(fill=tk.X)

        if TkinterDnD and DND_FILES:
            self.drop_area.drop_target_register(DND_FILES)
            self.drop_area.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self.drop_area.configure(text="Перетащите файл сюда\n(нужен пакет tkinterdnd2)")

        ttk.Button(parent, text="Выбрать файл", command=self._pick_file).pack(anchor="w", pady=12)

        ttk.Label(parent, text="Выбранный файл:").pack(anchor="w")
        ttk.Label(parent, textvariable=self.selected_file, wraplength=360).pack(anchor="w", pady=(4, 8))

        ttk.Label(parent, textvariable=self.status_text, foreground="red").pack(anchor="w")

    def _build_right(self, parent):
        ttk.Label(
            parent,
            text="Максимальные размеры детали",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 20))

        values_frame = ttk.Frame(parent)
        values_frame.pack(anchor="center")

        self._create_axis_block(values_frame, "X", "длина", self.x_value, 0)
        self._create_axis_block(values_frame, "Y", "ширина", self.y_value, 1)
        self._create_axis_block(values_frame, "Z", "высота", self.z_value, 2)

        # Площадь и объём
        extra_frame = ttk.Frame(parent)
        extra_frame.pack(anchor="center", pady=(30, 0))

        self._create_extra_block(extra_frame, "Площадь", "мм²", self.area_value, 0)
        self._create_extra_block(extra_frame, "Объём", "мм³", self.volume_value, 1)

        # Выбор единиц измерения
        unit_frame = ttk.Frame(parent)
        unit_frame.pack(anchor="center", pady=(20, 0))

        ttk.Label(unit_frame, text="Единицы измерения:").grid(row=0, column=0, padx=(0, 10))
        unit_combo = ttk.Combobox(unit_frame, textvariable=self.unit_var, values=["мм", "см", "м"], width=6, state="readonly")
        unit_combo.grid(row=0, column=1)
        unit_combo.bind("<<ComboboxSelected>>", self._update_units)

    def _create_axis_block(self, parent, axis, label, value_var, column):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(row=0, column=column, padx=12)

        ttk.Label(frame, text=axis, font=("Segoe UI", 12, "bold")).pack()
        ttk.Label(frame, text=label).pack()
        value_label = ttk.Label(frame, textvariable=value_var, relief="solid", width=10, anchor="center")
        value_label.pack(pady=6)
        unit_label = ttk.Label(frame, text="мм")
        unit_label.pack()
        self.unit_labels[f"axis_{column}"] = unit_label

    def _create_extra_block(self, parent, label, unit, value_var, column):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(row=0, column=column, padx=20)

        ttk.Label(frame, text=label, font=("Segoe UI", 11, "bold")).pack()
        value_label = ttk.Label(frame, textvariable=value_var, relief="solid", width=12, anchor="center")
        value_label.pack(pady=6)
        ttk.Label(frame, text=unit).pack()

    def _pick_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл детали",
            filetypes=[("DXF и STL", "*.dxf *.stl"), ("Все файлы", "*.*")],
        )
        if file_path:
            self._handle_file(file_path)

    def _on_drop(self, event):
        file_path = _extract_first_path(event.data, self.root)
        if file_path:
            self._handle_file(file_path)

    def _handle_file(self, file_path):
        self.selected_file.set(file_path)
        extension = os.path.splitext(file_path)[1].lower()

        if extension not in (".dxf", ".stl"):
            self.status_text.set("Неподходящий файл. Нужен .dxf или .stl")
            self._clear_raw()
            self._update_display()
            return

        self.status_text.set("")

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
            self.status_text.set(f"Ошибка чтения файла: {exc}")
            self._clear_raw()
            self._update_display()
            return

        if not dims:
            self.status_text.set("Не удалось вычислить габариты")
            self._clear_raw()
            self._update_display()
            return

        # Сохраняем сырые значения в мм
        self.raw_x, self.raw_y, self.raw_z = dims
        self.raw_area = area
        self.raw_volume = volume

        # Обновляем отображение с текущими единицами
        self._update_display()
        self._update_unit_labels()

    def _update_unit_labels(self):
        unit = self.unit_var.get()
        # Обновляем метки линейных размеров
        for i in range(3):
            key = f"axis_{i}"
            if key in self.unit_labels:
                self.unit_labels[key].config(text=unit)
        # Метки для площади и объёма уже статические, но можно было бы аналогично

    def _clear_raw(self):
        self.raw_x = None
        self.raw_y = None
        self.raw_z = None
        self.raw_area = None
        self.raw_volume = None

    def _update_units(self, event=None):
        self._update_display()
        self._update_unit_labels()

    def _update_display(self):
        unit = self.unit_var.get()

        # Линейные размеры
        if self.raw_x is not None:
            conv_x = _convert_value(self.raw_x, "мм", unit)
            self.x_value.set(_format_dimension(conv_x, unit))
        else:
            self.x_value.set("—")

        if self.raw_y is not None:
            conv_y = _convert_value(self.raw_y, "мм", unit)
            self.y_value.set(_format_dimension(conv_y, unit))
        else:
            self.y_value.set("—")

        if self.raw_z is not None:
            conv_z = _convert_value(self.raw_z, "мм", unit)
            self.z_value.set(_format_dimension(conv_z, unit))
        else:
            self.z_value.set("—")

        # Площадь
        if self.raw_area is not None:
            conv_area = _convert_value(self.raw_area, "мм", unit, is_square=True)
            self.area_value.set(_format_dimension(conv_area, unit))
        else:
            self.area_value.set("—")

        # Объём
        if self.raw_volume is not None:
            conv_volume = _convert_value(self.raw_volume, "мм", unit, is_cubic=True)
            self.volume_value.set(_format_dimension(conv_volume, unit))
        else:
            self.volume_value.set("—")


def main():
    root = TkinterDnD.Tk() if TkinterDnD else tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    BoundingBoxApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
