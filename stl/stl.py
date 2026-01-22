import struct

import numpy as np


def parse_stl_ascii(filename):
    """Разбор ASCII STL и возврат списка треугольников."""
    triangles = []

    with open(filename, "r") as f:
        current_triangle = {"normal": None, "vertices": []}

        for line in f:
            line = line.strip()

            if line.startswith("facet normal"):
                parts = line.split()
                current_triangle["normal"] = [float(parts[2]), float(parts[3]), float(parts[4])]

            elif line.startswith("vertex"):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                current_triangle["vertices"].append(vertex)

            elif line.startswith("endfacet"):
                if len(current_triangle["vertices"]) == 3:
                    triangles.append(current_triangle)
                current_triangle = {"normal": None, "vertices": []}

    return triangles


def parse_stl_binary(filename):
    """Разбор бинарного STL и возврат списка треугольников."""
    triangles = []

    with open(filename, "rb") as f:
        # Заголовок 80 байт
        f.read(80)

        # Количество треугольников (4-байтовое беззнаковое целое)
        num_triangles = struct.unpack("I", f.read(4))[0]

        # Чтение каждого треугольника
        for _ in range(num_triangles):
            # Нормаль (3 float = 12 байт)
            normal = struct.unpack("fff", f.read(12))

            # Три вершины (3 * 3 float = 36 байт)
            v1 = struct.unpack("fff", f.read(12))
            v2 = struct.unpack("fff", f.read(12))
            v3 = struct.unpack("fff", f.read(12))

            # Атрибуты (2 байта, обычно не используются)
            f.read(2)

            triangles.append({
                "normal": list(normal),
                "vertices": [list(v1), list(v2), list(v3)],
            })
    return triangles


def is_binary_stl(filename):
    """Определение формата STL (бинарный или ASCII)."""
    try:
        with open(filename, "r") as f:
            first_line = f.read(100)
            if "solid" in first_line.lower():
                return False
    except UnicodeDecodeError:
        return True

    # Дополнительная проверка: бинарный формат имеет точный размер
    import os

    file_size = os.path.getsize(filename)
    with open(filename, "rb") as f:
        f.read(80)
        num_triangles = struct.unpack("I", f.read(4))[0]
        expected_size = 80 + 4 + num_triangles * 50
        if file_size == expected_size:
            return True

    return False


def parse_stl(filename):
    """Разбор STL (автоопределение ASCII или бинарного формата)."""
    if is_binary_stl(filename):
        return parse_stl_binary(filename)
    return parse_stl_ascii(filename)


def calculate_triangle_area(triangle):
    """
    Площадь треугольника по формуле векторного произведения.
    Area = 0.5 * ||(v2 - v1) × (v3 - v1)||
    """
    v1, v2, v3 = [np.array(v) for v in triangle["vertices"]]

    # Векторы ребер
    edge1 = v2 - v1
    edge2 = v3 - v1

    # Векторное произведение
    cross = np.cross(edge1, edge2)

    # Площадь равна половине длины векторного произведения
    area = 0.5 * np.linalg.norm(cross)

    return area


def find_max_area_triangle(filename):
    """Поиск треугольника с максимальной площадью в STL."""
    triangles = parse_stl(filename)

    if not triangles:
        return None

    max_area = 0.0
    max_triangle = None
    max_index = -1
    cnt_triangles = len(triangles)

    for i, triangle in enumerate(triangles):
        area = calculate_triangle_area(triangle)

        if area > max_area:
            max_area = area
            max_triangle = triangle
            max_index = i

    return {
        "count" : cnt_triangles,
        "index": max_index,
        "area": max_area,
        "triangle": max_triangle,
    }


def create_coordinate_system_from_triangle(triangle):
    """
    Формирование ортонормированной СК по плоскости треугольника.
    Возвращает матрицу вращения, где:
    - ось Z перпендикулярна треугольнику (нормаль)
    - ось X направлена вдоль одного ребра
    - ось Y дополняет правую систему координат
    """
    v1, v2, v3 = [np.array(v) for v in triangle["vertices"]]

    # Векторы ребер
    edge1 = v2 - v1
    edge2 = v3 - v1

    # Ось Z: нормаль к плоскости
    z_axis = np.cross(edge1, edge2)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Ось X: вдоль первого ребра
    x_axis = edge1 / np.linalg.norm(edge1)

    # Ось Y: перпендикуляр к X и Z
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Матрица вращения (каждый столбец — базисный вектор)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    return rotation_matrix, v1


def calculate_aligned_bounding_box(triangles, normal_vector, origin):
    """
    Минимальный бокс, у которого одна грань перпендикулярна normal_vector.
    Ищется оптимальный поворот в плоскости.
    """
    # Уникальные вершины для снижения дубликатов
    unique_vertices = set()
    for triangle in triangles:
        for vertex in triangle["vertices"]:
            vertex_tuple = tuple(np.round(vertex, decimals=10))
            unique_vertices.add(vertex_tuple)

    all_vertices = [np.array(v) for v in unique_vertices]

    # Нормализуем нормаль
    z_axis = normal_vector / np.linalg.norm(normal_vector)

    # Проекция вершин на плоскость перпендикулярную нормали
    projected_2d = []
    for vertex in all_vertices:
        v_translated = vertex - origin
        projection_length = np.dot(v_translated, z_axis)
        v_in_plane = v_translated - projection_length * z_axis
        projected_2d.append(v_in_plane)

    min_area = float("inf")
    best_rotation = None

    # Пробуем поворот с шагом 1 градус
    for angle_deg in np.arange(0, 180, 0.1):
        angle_rad = np.deg2rad(angle_deg)

        # Базовые оси в плоскости
        temp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])

        x_base = np.cross(z_axis, temp)
        x_base = x_base / np.linalg.norm(x_base)
        y_base = np.cross(z_axis, x_base)
        y_base = y_base / np.linalg.norm(y_base)

        # Поворот в плоскости
        x_axis = np.cos(angle_rad) * x_base + np.sin(angle_rad) * y_base
        y_axis = -np.sin(angle_rad) * x_base + np.cos(angle_rad) * y_base

        # Проекция всех точек на оси
        x_coords = [np.dot(v, x_axis) for v in projected_2d]
        y_coords = [np.dot(v, y_axis) for v in projected_2d]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        area = x_range * y_range

        if area < min_area:
            min_area = area
            best_rotation = np.column_stack([x_axis, y_axis, z_axis])

    # Итоговый бокс в найденной ориентации
    transformed_vertices = []
    for vertex in all_vertices:
        translated = vertex - origin
        transformed = best_rotation.T @ translated
        transformed_vertices.append(transformed)

    transformed_vertices = np.array(transformed_vertices)

    min_coords = np.min(transformed_vertices, axis=0)
    max_coords = np.max(transformed_vertices, axis=0)

    return min_coords, max_coords, best_rotation


def calculate_nonrotated_bounding_box(triangles):
    """Осеориентированный бокс (без вращения)."""
    unique_vertices = set()
    for triangle in triangles:
        for vertex in triangle["vertices"]:
            vertex_tuple = tuple(np.round(vertex, decimals=10))
            unique_vertices.add(vertex_tuple)

    all_vertices = np.array([list(v) for v in unique_vertices])

    min_coords = np.min(all_vertices, axis=0)
    max_coords = np.max(all_vertices, axis=0)

    return min_coords, max_coords


def calculate_parallelepiped_volume(filename):
    """
    Объём минимального параллелепипеда, ориентированного по плоскости
    треугольника максимальной площади.
    """
    triangles = parse_stl(filename)

    if not triangles:
        return None

    max_area = 0.0
    max_triangle = None
    max_index = -1

    for i, triangle in enumerate(triangles):
        area = calculate_triangle_area(triangle)

        if area > max_area:
            max_area = area
            max_triangle = triangle
            max_index = i

    v1, v2, v3 = [np.array(v) for v in max_triangle["vertices"]]
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal_vector = np.cross(edge1, edge2)

    min_coords, max_coords, rotation_matrix = calculate_aligned_bounding_box(triangles, normal_vector, v1)

    dimensions = max_coords - min_coords

    volume = dimensions[0] * dimensions[1] * dimensions[2]

    return {
        "max_triangle_index": max_index,
        "max_triangle_area": max_area,
        "max_triangle": max_triangle,
        "min_coords": min_coords,
        "max_coords": max_coords,
        "dimensions": dimensions,
        "volume": volume,
        "rotation_matrix": rotation_matrix,
        "origin": v1,
    }


def calculate_nonrotated_volume(filename):
    """Объём осеориентированного бокса (без вращения)."""
    triangles = parse_stl(filename)

    if not triangles:
        return None

    min_coords, max_coords = calculate_nonrotated_bounding_box(triangles)

    dimensions = max_coords - min_coords

    volume = dimensions[0] * dimensions[1] * dimensions[2]

    return {
        "min_coords": min_coords,
        "max_coords": max_coords,
        "dimensions": dimensions,
        "volume": volume,
    }


def get_bounding_box_dimensions(filename, *, aligned=True):
    """Вернуть размеры габаритного бокса (X, Y, Z) для STL."""
    if aligned:
        result = calculate_parallelepiped_volume(filename)
    else:
        result = calculate_nonrotated_volume(filename)

    if not result:
        return None

    dims = result["dimensions"]
    return dims[0], dims[1], dims[2]


def main():
    import sys

    if len(sys.argv) < 2:
        print("Использование: python stl.py <stl_file> [опции]")
        print("Опции:")
        print("  --volume          Рассчитать объём ориентированного параллелепипеда")
        print("  --nonrot-volume   Рассчитать объём осеориентированного бокса")
        print("\nПримеры:")
        print("  python stl.py model.stl")
        print("  python stl.py model.stl --volume")
        print("  python stl.py model.stl --nonrot-volume")
        return

    filename = sys.argv[1]
    calculate_volume = "--volume" in sys.argv
    calculate_nonrot_volume = "--nonrot-volume" in sys.argv

    try:
        if calculate_nonrot_volume:
            result = calculate_nonrotated_volume(filename)

            if result:
                print("\n" + "=" * 60)
                print("Осеориентированный бокс (без вращения)")
                print("=" * 60)
                print("\nРазмеры бокса:")
                print(
                    "  Габариты (X, Y, Z): "
                    f"[{result['dimensions'][0]:.6f}, {result['dimensions'][1]:.6f}, {result['dimensions'][2]:.6f}]"
                )
                print(
                    "  Минимальный угол: "
                    f"[{result['min_coords'][0]:.6f}, {result['min_coords'][1]:.6f}, {result['min_coords'][2]:.6f}]"
                )
                print(
                    "  Максимальный угол: "
                    f"[{result['max_coords'][0]:.6f}, {result['max_coords'][1]:.6f}, {result['max_coords'][2]:.6f}]"
                )
                print(f"\n  ОБЪЁМ: {result['volume']:.6f}")
                print("=" * 60)
        elif calculate_volume:
            result = calculate_parallelepiped_volume(filename)

            if result:
                print("\n" + "=" * 60)
                print("Ориентированный параллелепипед")
                print("=" * 60)
                print("\nТреугольник максимальной площади:")
                print(f"  Индекс: {result['max_triangle_index']}")
                print(f"  Площадь: {result['max_triangle_area']:.6f}")
                print("\nГабариты параллелепипеда:")
                print(
                    "  Размеры (X, Y, Z): "
                    f"[{result['dimensions'][0]:.6f}, {result['dimensions'][1]:.6f}, {result['dimensions'][2]:.6f}]"
                )
                print(
                    "  Минимальный угол: "
                    f"[{result['min_coords'][0]:.6f}, {result['min_coords'][1]:.6f}, {result['min_coords'][2]:.6f}]"
                )
                print(
                    "  Максимальный угол: "
                    f"[{result['max_coords'][0]:.6f}, {result['max_coords'][1]:.6f}, {result['max_coords'][2]:.6f}]"
                )
                print(f"\n  ОБЪЁМ: {result['volume']:.6f}")
                print("=" * 60)
        else:
            result = find_max_area_triangle(filename)

            if result:
                print("\n" + "=" * 60)
                print("Треугольник максимальной площади")
                print("=" * 60)
                print(f"Количество треугольников: {result['count']}")
                print(f"Индекс: {result['index']}")
                print(f"Площадь: {result['area']:.6f}")
                print(f"\nНормаль: {result['triangle']['normal']}")
                print("\nВершины:")
                for i, vertex in enumerate(result["triangle"]["vertices"], 1):
                    print(f"  Вершина {i}: {vertex}")
                print("=" * 60)

    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден")
    except Exception as exc:
        print(f"Ошибка: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
