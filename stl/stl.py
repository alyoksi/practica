import struct
import numpy as np
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def _is_binary_stl_fast(filename):
    """Быстрое определение бинарного STL."""
    import os
    file_size = os.path.getsize(filename)
    
    # Быстрая проверка: если размер файла < 84 байт, не может быть бинарным
    if file_size < 84:
        return False

    with open(filename, 'rb') as f:
        header = f.read(80)
        # Проверка, содержит ли заголовок не-ASCII символы
        try:
            header.decode('ascii')
            # Если декодируется успешно, проверка на 'solid' в начале
            if header[:5].lower() == b'solid':
                # Может быть бинарным, если размер соответствует бинарному формату
                pass
        except UnicodeDecodeError:
            return True

        # Проверка бинарной структуры
        f.seek(80)
        num_triangles_bytes = f.read(4)
        if len(num_triangles_bytes) < 4:
            return False
        num_triangles = struct.unpack('I', num_triangles_bytes)[0]
        expected_size = 80 + 4 + num_triangles * 50
        return file_size == expected_size


def _parse_stl_binary_streaming(filename, progress_callback=None, chunk_size=500000):
    """
    Разбор бинарного STL с потоковой обработкой для больших файлов.
    Возвращает: (max_area_info, unique_vertices, total_triangles)
    где max_area_info = (max_area, max_index, max_normal, max_vertices)
    """
    import os
    
    file_size = os.path.getsize(filename)
    
    with open(filename, 'rb') as f:
        # Чтение заголовка и количества треугольников
        f.read(80)  # Пропустить заголовок
        num_triangles_bytes = f.read(4)
        num_triangles = struct.unpack('I', num_triangles_bytes)[0]

    # Отображение файла в память для доступа без копирования
    mmap = np.memmap(filename, dtype=np.uint8, mode='r')

    # Пропустить заголовок (80 байт) и количество треугольников (4 байта)
    data_offset = 84

    # Инициализация переменных отслеживания
    max_area = 0.0
    max_index = -1
    max_normal = None
    max_vertices = None

    # Сбор вершин порциями
    vertex_chunks = []

    # Обработка порциями для контроля использования памяти
    for chunk_start in range(0, num_triangles, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_triangles)
        chunk_size_current = chunk_end - chunk_start

        # Вычисление байтовых смещений для этой порции
        chunk_byte_start = data_offset + chunk_start * 50
        chunk_byte_end = data_offset + chunk_end * 50

        # Извлечение данных порции в виде структурированного массива - намного быстрее
        chunk_data = np.frombuffer(
            mmap[chunk_byte_start:chunk_byte_end],
            dtype=np.dtype([
                ('normal', '3f4'),
                ('v1', '3f4'),
                ('v2', '3f4'),
                ('v3', '3f4'),
                ('attr', 'u2')
            ])
        )

        # Извлечение всех вершин из этой порции (chunk_size_current * 3 вершины)
        # Преобразование для получения всех вершин в виде массива (chunk_size_current * 3, 3)
        vertices_chunk = np.column_stack([
            chunk_data['v1'].reshape(-1, 1),
            chunk_data['v1'].reshape(-1, 1),
            chunk_data['v1'].reshape(-1, 1)
        ])

        # На самом деле, нужно получить v1, v2, v3 отдельно
        # Создание массива со всеми вершинами из этой порции
        vertices_all = np.zeros((chunk_size_current * 3, 3), dtype=np.float32)
        vertices_all[0::3] = chunk_data['v1']
        vertices_all[1::3] = chunk_data['v2']
        vertices_all[2::3] = chunk_data['v3']

        # Округление вершин для удаления дубликатов и добавление в порции
        vertices_rounded = np.round(vertices_all, 6)
        vertex_chunks.append(vertices_rounded)

        # Обработка треугольников для поиска максимальной площади (векторно)
        # Вычисление площадей для всех треугольников в этой порции
        v1 = chunk_data['v1']
        v2 = chunk_data['v2']
        v3 = chunk_data['v3']

        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        # Поиск максимальной площади в этой порции
        chunk_max_idx = np.argmax(areas)
        chunk_max_area = areas[chunk_max_idx]

        # Обновление глобального максимума
        if chunk_max_area > max_area:
            max_area = chunk_max_area
            max_index = chunk_start + chunk_max_idx
            max_normal = chunk_data[chunk_max_idx]['normal'].copy()
            max_vertices = np.array([
                v1[chunk_max_idx],
                v2[chunk_max_idx],
                v3[chunk_max_idx]
            ])

        # Сообщение о прогрессе
        if progress_callback:
            progress = (chunk_end / num_triangles) * 100
            progress_callback(progress, f"Processing triangles: {chunk_end:,}/{num_triangles:,}")
    
    # Объединение и удаление дубликатов вершин
    if vertex_chunks:
        # Конкатенация всех вершин
        all_vertices = np.concatenate(vertex_chunks, axis=0)
        # Использование numpy unique для удаления дубликатов (намного быстрее, чем Python set)
        unique_vertices = np.unique(all_vertices, axis=0)
    else:
        unique_vertices = np.zeros((0, 3), dtype=np.float32)

    # Создание словаря с информацией о максимальном треугольнике
    max_triangle_info = {
        'max_area': max_area,
        'max_index': max_index,
        'max_normal': max_normal,
        'max_vertices': max_vertices,
    }

    return max_triangle_info, unique_vertices, num_triangles


def _calculate_triangle_area_from_vertices(v1, v2, v3):
    """Вычисление площади треугольника по трем вершинам."""
    edge1 = v2 - v1
    edge2 = v3 - v1
    cross = np.cross(edge1, edge2)
    return 0.5 * np.linalg.norm(cross)


def calculate_parallelepiped_volume_streaming(filename, progress_callback=None):
    """
    Экономная версия для больших STL файлов (>1M треугольников).
    Использует потоковый разбор с отображением в память.
    """
    import numpy as np
    import time
    
    if progress_callback:
        progress_callback(0, "Starting STL processing...")

    start_time = time.time()

    # Использование потокового парсера
    max_info, unique_vertices, total_triangles = _parse_stl_binary_streaming(
        filename, progress_callback=progress_callback
    )

    if progress_callback:
        progress_callback(50, "Finding bounding box...")

    # Извлечение информации о максимальном треугольнике
    max_area = max_info['max_area']
    max_index = max_info['max_index']
    max_normal = max_info['max_normal']
    max_vertices = max_info['max_vertices']

    # Создание системы координат из максимального треугольника
    v1, v2, v3 = max_vertices
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal_vector = np.cross(edge1, edge2)
    origin = v1

    # Вычисление выровненного ограничивающего бокса с использованием уникальных вершин
    if len(unique_vertices) == 0:
        return None

    # Использование оптимизированного вычисления ограничивающего бокса
    min_coords, max_coords, rotation_matrix = calculate_aligned_bounding_box_optimized_large(
        unique_vertices, normal_vector, origin
    )

    dimensions = max_coords - min_coords
    volume = dimensions[0] * dimensions[1] * dimensions[2]

    # Создание словаря максимального треугольника для совместимости
    max_triangle_dict = {
        'normal': max_normal.tolist(),
        'vertices': max_vertices.tolist()
    }

    end_time = time.time()

    if progress_callback:
        progress_callback(100, f"Completed in {end_time - start_time:.1f} seconds")
    
    return {
        'total_triangles': total_triangles,
        'max_triangle_index': max_index,
        'max_triangle_area': max_area,
        'max_triangle': max_triangle_dict,
        'min_coords': min_coords,
        'max_coords': max_coords,
        'dimensions': dimensions,
        'volume': volume,
        'rotation_matrix': rotation_matrix,
        'origin': origin.tolist(),
    }


def calculate_aligned_bounding_box_optimized_large(unique_vertices, normal_vector, origin):
    """
    Оптимизированное вычисление ограничивающего бокса для больших наборов вершин.
    Использует приближение выпуклой оболочки для скорости.
    """
    if len(unique_vertices) == 0:
        return np.zeros(3), np.zeros(3), np.eye(3)
    
    # Нормализация нормали
    z_axis = normal_vector / np.linalg.norm(normal_vector)

    # Проекция вершин на плоскость, перпендикулярную нормали
    v_translated = unique_vertices - origin
    projection_lengths = np.dot(v_translated, z_axis)

    # Использование трансляции для эффективности
    projection_lengths_3d = projection_lengths[:, np.newaxis]
    projected_2d = v_translated - projection_lengths_3d * z_axis

    # Для очень больших наборов, выборка для выпуклой оболочки
    if len(projected_2d) > 10000:
        # Использование случайной выборки для приближения выпуклой оболочки
        sample_indices = np.random.choice(len(projected_2d), size=10000, replace=False)
        hull_points_2d = compute_convex_hull_2d(projected_2d[sample_indices, :2])
    else:
        hull_points_2d = compute_convex_hull_2d(projected_2d[:, :2])

    if len(hull_points_2d) == 0:
        return np.zeros(3), np.zeros(3), np.eye(3)

    # Поиск минимального прямоугольника с использованием вращающихся штангенциркулей
    min_area, angle, width, height = rotating_calipers_min_area_rectangle(hull_points_2d)

    # Создание матрицы вращения для оптимальной ориентации
    temp = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_base = np.cross(z_axis, temp)
    x_base = x_base / np.linalg.norm(x_base)
    y_base = np.cross(z_axis, x_base)
    y_base = y_base / np.linalg.norm(y_base)

    # Оптимальное вращение в плоскости
    x_axis = np.cos(angle) * x_base + np.sin(angle) * y_base
    y_axis = -np.sin(angle) * x_base + np.cos(angle) * y_base

    # Итоговая матрица вращения
    best_rotation = np.column_stack([x_axis, y_axis, z_axis])

    # Трансформация всех вершин для нахождения экстентов
    transformed = np.dot(unique_vertices - origin, best_rotation)
    min_coords = transformed.min(axis=0)
    max_coords = transformed.max(axis=0)
    
    return min_coords, max_coords, best_rotation


def get_bounding_box_dimensions_streaming(filename, *, aligned=True, progress_callback=None):
    """Потоковая версия get_bounding_box_dimensions."""
    if not aligned:
        # Для невращаемого бокса
        max_info, unique_vertices, total_triangles = _parse_stl_binary_streaming(
            filename, progress_callback=progress_callback
        )
        if len(unique_vertices) == 0:
            return None
        min_coords = unique_vertices.min(axis=0)
        max_coords = unique_vertices.max(axis=0)
        dimensions = max_coords - min_coords
        return dimensions[0], dimensions[1], dimensions[2]
    else:
        result = calculate_parallelepiped_volume_streaming(filename, progress_callback)
        if not result:
            return None
        dims = result['dimensions']
        return dims[0], dims[1], dims[2]


def _parse_stl_binary_vectorized(filename):
    """Разбор бинарного STL с векторизованным выводом."""
    with open(filename, 'rb') as f:
        f.read(80)  # Пропустить заголовок
        num_triangles = struct.unpack('I', f.read(4))[0]

        # Предварительное выделение массивов
        vertices = np.zeros((num_triangles, 3, 3), dtype=np.float32)
        normals = np.zeros((num_triangles, 3), dtype=np.float32)

        for i in range(num_triangles):
            # Чтение нормали (3 float)
            normal = struct.unpack('fff', f.read(12))
            normals[i] = normal

            # Чтение вершин (9 float)
            v1 = struct.unpack('fff', f.read(12))
            v2 = struct.unpack('fff', f.read(12))
            v3 = struct.unpack('fff', f.read(12))
            vertices[i, 0] = v1
            vertices[i, 1] = v2
            vertices[i, 2] = v3

            # Пропуск байтов атрибутов
            f.read(2)

        # Создание словарей для совместимости
        triangles = []
        for i in range(num_triangles):
            triangles.append({
                'normal': normals[i].tolist(),
                'vertices': vertices[i].tolist()
            })

        return vertices, triangles


def _parse_stl_ascii_vectorized(filename):
    """Разбор ASCII STL с векторизованным выводом."""
    vertices_list = []
    normals_list = []
    triangles = []
    
    with open(filename, 'r') as f:
        current_vertices = []
        current_normal = None
        
        for line in f:
            line = line.strip()
            
            if line.startswith('facet normal'):
                parts = line.split()
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                current_vertices = []
                
            elif line.startswith('vertex'):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                current_vertices.append(vertex)
                
            elif line.startswith('endfacet'):
                if len(current_vertices) == 3:
                    vertices_list.append(current_vertices)
                    normals_list.append(current_normal)
                    triangles.append({
                        'normal': current_normal,
                        'vertices': current_vertices.copy()
                    })
                current_vertices = []
                current_normal = None
    
    if vertices_list:
        vertices = np.array(vertices_list, dtype=np.float32)
    else:
        vertices = np.zeros((0, 3, 3), dtype=np.float32)
    
    return vertices, triangles


def parse_stl_vectorized(filename):
    """
    Разбор STL файла и возврат вершин в виде numpy массивов.
    Возвращает: (vertices, triangles) где:
        vertices: (n, 3, 3) массив вершин треугольников
        triangles: список оригинальных словарей для совместимости
    """
    import time
    start_time = time.time()

    file_output = _parse_stl_binary_vectorized(filename) if _is_binary_stl_fast(filename) else _parse_stl_ascii_vectorized(filename)

    print("---Парсинг: %s секунд ---" % (time.time() - start_time))
    return file_output


def calculate_triangle_areas_vectorized(vertices):
    """
    Вычисление площадей треугольников с использованием векторизованных операций.
    vertices: (n, 3, 3) массив где vertices[i, j] - это j-я вершина i-го треугольника
    Возвращает: (n,) массив площадей
    """
    # Векторы ребер
    v1 = vertices[:, 0]  # (n, 3)
    v2 = vertices[:, 1]  # (n, 3)
    v3 = vertices[:, 2]  # (n, 3)

    edge1 = v2 - v1  # (n, 3)
    edge2 = v3 - v1  # (n, 3)

    # Векторное произведение
    cross = np.cross(edge1, edge2)  # (n, 3)

    # Площадь = 0.5 * норма векторного произведения
    areas = 0.5 * np.linalg.norm(cross, axis=1)  # (n,)

    return areas


def find_max_area_triangle_vectorized(vertices):
    """
    Поиск треугольника с максимальной площадью.
    Возвращает: (max_area, max_index, max_triangle_vertices)
    """
    if vertices.shape[0] == 0:
        return 0.0, -1, None
    
    areas = calculate_triangle_areas_vectorized(vertices)
    max_index = np.argmax(areas)
    max_area = areas[max_index]
    max_triangle_vertices = vertices[max_index]
    
    return max_area, max_index, max_triangle_vertices


def create_coordinate_system_from_triangle_vectorized(triangle_vertices):
    """
    Создание ортонормированной системы координат из треугольника.
    triangle_vertices: (3, 3) массив вершин
    Возвращает: (rotation_matrix, origin)
    """
    v1, v2, v3 = triangle_vertices

    # Векторы ребер
    edge1 = v2 - v1
    edge2 = v3 - v1

    # Ось Z: нормаль к плоскости
    z_axis = np.cross(edge1, edge2)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-10:
        # Вырожденный треугольник, использовать оси по умолчанию
        return np.eye(3), v1

    z_axis = z_axis / z_norm

    # Ось X: вдоль первого ребра
    x_axis = edge1
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-10:
        # Ребро слишком короткое, использовать произвольный перпендикулярный вектор
        if abs(z_axis[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, temp)
        x_norm = np.linalg.norm(x_axis)

    x_axis = x_axis / x_norm

    # Ось Y: перпендикулярна X и Z
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-10:
        # Не должно происходить при правильной нормализации
        y_axis = np.array([0.0, 0.0, 1.0])
    else:
        y_axis = y_axis / y_norm

    # Матрица вращения (столбцы - базисные векторы)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    return rotation_matrix, v1


def compute_convex_hull_2d(points):
    """
    Вычисление 2D выпуклой оболочки точек с использованием SciPy.
    points: (n, 2) массив 2D точек
    Возвращает: hull_points (m, 2) массив вершин оболочки в порядке против часовой стрелки
    """
    if len(points) < 3:
        return points

    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except:
        # Резервный вариант: вернуть оригинальные точки, если оболочка не удалась
        return points


def rotating_calipers_min_area_rectangle(points):
    """
    Поиск прямоугольника минимальной площади с использованием алгоритма вращающихся штангенциркулей.
    points: (n, 2) массив точек выпуклой оболочки в порядке против часовой стрелки
    Возвращает: (min_area, rotation_angle, width, height)
    """
    n = len(points)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0

    # Инициализация штангенциркулей
    min_area = float('inf')
    best_angle = 0.0
    best_width = 0.0
    best_height = 0.0

    # Для каждого ребра как базового направления
    for i in range(n):
        # Текущее направление ребра
        p1 = points[i]
        p2 = points[(i + 1) % n]
        edge_dir = p2 - p1
        edge_len = np.linalg.norm(edge_dir)

        if edge_len < 1e-10:
            continue

        # Единичный вектор направления
        u = edge_dir / edge_len

        # Перпендикулярный вектор
        v = np.array([-u[1], u[0]])

        # Проекция всех точек на оси u и v
        proj_u = np.dot(points, u)
        proj_v = np.dot(points, v)

        # Вычисление экстентов
        u_min, u_max = proj_u.min(), proj_u.max()
        v_min, v_max = proj_v.min(), proj_v.max()

        width = u_max - u_min
        height = v_max - v_min
        area = width * height

        if area < min_area:
            min_area = area
            best_angle = np.arctan2(u[1], u[0])
            best_width = width
            best_height = height

    return min_area, best_angle, best_width, best_height


def calculate_aligned_bounding_box_optimized(vertices, normal_vector, origin):
    """
    Оптимизированное вычисление ограничивающего бокса с использованием выпуклой оболочки и вращающихся штангенциркулей.
    Возвращает размеры, отсортированные по размеру: (наибольший, средний, наименьший)
    """
    if vertices.shape[0] == 0:
        return np.zeros(3), np.zeros(3), np.eye(3)

    # Нормализация нормали
    z_axis = normal_vector / np.linalg.norm(normal_vector)

    # Извлечение всех уникальных вершин
    # Преобразование в (n*3, 3) и использование unique
    all_vertices = vertices.reshape(-1, 3)

    # Использование округления для уменьшения дубликатов
    rounded_vertices = np.round(all_vertices, decimals=6)
    unique_vertices = np.unique(rounded_vertices, axis=0)

    # Проекция вершин на плоскость, перпендикулярную нормали
    v_translated = unique_vertices - origin
    projection_lengths = np.dot(v_translated, z_axis)

    projection_lengths_3d = projection_lengths[:, np.newaxis]
    projected_3d = v_translated - projection_lengths_3d * z_axis

    # Создание ортонормированного базиса для плоскости
    # Выбор произвольного вектора, перпендикулярного z_axis для x_base
    if abs(z_axis[0]) < 0.9:
        temp = np.array([1.0, 0.0, 0.0])
    else:
        temp = np.array([0.0, 1.0, 0.0])
    x_base = np.cross(z_axis, temp)
    x_base = x_base / np.linalg.norm(x_base)
    y_base = np.cross(z_axis, x_base)
    y_base = y_base / np.linalg.norm(y_base)

    # Проекция 3D точек на 2D плоскость с использованием базиса плоскости
    x_coords = np.dot(projected_3d, x_base)
    y_coords = np.dot(projected_3d, y_base)
    projected_2d = np.column_stack([x_coords, y_coords])

    # Вычисление 2D выпуклой оболочки
    hull_points_2d = compute_convex_hull_2d(projected_2d)

    if len(hull_points_2d) == 0:
        return np.zeros(3), np.zeros(3), np.eye(3)

    # Поиск минимального прямоугольника с использованием алг. вращающихся калиперов
    min_area, angle, width, height = rotating_calipers_min_area_rectangle(hull_points_2d)

    # Создание матрицы вращения для оптимальной ориентации в плоскости
    x_axis = np.cos(angle) * x_base + np.sin(angle) * y_base
    y_axis = -np.sin(angle) * x_base + np.cos(angle) * y_base

    # Итоговая матрица вращения
    best_rotation = np.column_stack([x_axis, y_axis, z_axis])

    # Трансформация всех вершин для нахождения экстентов
    transformed = np.dot(unique_vertices - origin, best_rotation)
    min_coords = transformed.min(axis=0)
    max_coords = transformed.max(axis=0)
    dimensions = max_coords - min_coords

    # Сортировка размеров по размеру (наибольший первым) для обеспечения согласованного порядка
    sorted_indices = np.argsort(-np.abs(dimensions))  # Отрицательное для убывания
    dimensions_sorted = dimensions[sorted_indices]
    min_coords_sorted = min_coords[sorted_indices]
    max_coords_sorted = max_coords[sorted_indices]

    # Также переупорядочивание столбцов матрицы вращения для соответствия отсортированным размерам
    rotation_sorted = best_rotation[:, sorted_indices]

    return min_coords_sorted, max_coords_sorted, rotation_sorted


def parse_stl_ascii(filename):
    """Разбор ASCII STL и возврат списка треугольников."""
    _, triangles = parse_stl_vectorized(filename)
    return triangles


def parse_stl_binary(filename):
    """Разбор бинарного STL и возврат списка треугольников."""
    _, triangles = parse_stl_vectorized(filename)
    return triangles


def is_binary_stl(filename):
    """Определение формата STL (бинарный или ASCII)."""
    return _is_binary_stl_fast(filename)


def parse_stl(filename):
    """Разбор STL (автоопределение ASCII или бинарного формата)."""
    _, triangles = parse_stl_vectorized(filename)
    return triangles


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
    vertices, triangles = parse_stl_vectorized(filename)

    if vertices.shape[0] == 0:
        return None

    max_area, max_index, _ = find_max_area_triangle_vectorized(vertices)

    return {
        "count": vertices.shape[0],
        "index": max_index,
        "area": max_area,
        "triangle": triangles[max_index],
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
    for angle_deg in np.arange(0, 180, 1):
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
    # Использование оптимизированной реализации
    vertices, triangles = parse_stl_vectorized(filename)
    total_triangles = vertices.shape[0]

    if total_triangles == 0:
        return None

    # Поиск треугольника с максимальной площадью
    max_area, max_index, max_triangle_vertices = find_max_area_triangle_vectorized(vertices)

    # Создание системы координат из максимального треугольника
    v1, v2, v3 = max_triangle_vertices
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal_vector = np.cross(edge1, edge2)

    # Вычисление выровненного ограничивающего бокса (оптимизированно)
    min_coords, max_coords, rotation_matrix = calculate_aligned_bounding_box_optimized(
        vertices, normal_vector, v1
    )

    dimensions = max_coords - min_coords
    volume = dimensions[0] * dimensions[1] * dimensions[2]

    max_triangle_dict = triangles[max_index]

    return {
        "total_triangles": total_triangles,
        "max_triangle_index": max_index,
        "max_triangle_area": max_area,
        "max_triangle": max_triangle_dict,
        "min_coords": min_coords,
        "max_coords": max_coords,
        "dimensions": dimensions,
        "volume": volume,
        "rotation_matrix": rotation_matrix,
        "origin": v1.tolist(),
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
