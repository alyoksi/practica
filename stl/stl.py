import struct
import numpy as np
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import collections
import math


def _edge_plane_intersections(va, vb, da, db, eps):
    """Вычислить точки пересечения ребра с плоскостью для массива рёбер."""
    near_a = np.abs(da) <= eps
    near_b = np.abs(db) <= eps
    edge_mask = (da * db < 0) | near_a | near_b
    edge_mask &= ~(near_a & near_b)
    denom = da - db
    valid = edge_mask & (np.abs(denom) > eps)
    t = np.zeros_like(da)
    t[valid] = da[valid] / denom[valid]
    pts = va + (vb - va) * t[:, np.newaxis]
    pts_2d = pts[:, :2]
    pts_2d[~edge_mask] = np.nan
    return pts_2d


def _slice_triangle_arrays(v1, v2, v3, plane_z, eps, grid_step):
    """Сегменты пересечения для массивов треугольников."""
    if len(v1) == 0:
        return []

    d1 = v1[:, 2] - plane_z
    d2 = v2[:, 2] - plane_z
    d3 = v3[:, 2] - plane_z

    p12 = _edge_plane_intersections(v1, v2, d1, d2, eps)
    p23 = _edge_plane_intersections(v2, v3, d2, d3, eps)
    p31 = _edge_plane_intersections(v3, v1, d3, d1, eps)

    valid12 = ~np.isnan(p12[:, 0])
    valid23 = ~np.isnan(p23[:, 0])
    valid31 = ~np.isnan(p31[:, 0])
    counts = valid12.astype(np.int32) + valid23.astype(np.int32) + valid31.astype(np.int32)
    indices = np.where(counts >= 2)[0]

    segments = []
    for idx in indices:
        pts = []
        if valid12[idx]:
            pts.append(p12[idx])
        if valid23[idx]:
            pts.append(p23[idx])
        if valid31[idx]:
            pts.append(p31[idx])

        keys = []
        for pt in pts:
            key = (int(np.rint(pt[0] / grid_step)), int(np.rint(pt[1] / grid_step)))
            if key not in keys:
                keys.append(key)

        if len(keys) < 2:
            continue
        if len(keys) > 2:
            max_d = -1
            best_pair = None
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    dx = keys[i][0] - keys[j][0]
                    dy = keys[i][1] - keys[j][1]
                    dist = dx * dx + dy * dy
                    if dist > max_d:
                        max_d = dist
                        best_pair = (keys[i], keys[j])
            if best_pair:
                segments.append(best_pair)
        else:
            segments.append((keys[0], keys[1]))

    return segments


def slice_stl_plane_segments_binary(filename, z_plane=0.0, eps=1e-6, delta=0.01,
                                    grid_step=0.2, chunk_size=500000, progress_callback=None,
                                    max_segments=5_000_000, rotation_matrix=None,
                                    origin=None, plane_in_aligned=False):
    """
    Потоковое вычисление отрезков пересечения STL с плоскостью Z=z_plane.
    Возвращает: (segments, total_triangles, total_segments)
    segments: список пар ((x1,y1),(x2,y2)) в виде целых ключей сетки.
    """
    plane_z = float(z_plane) + float(delta)

    segments = []
    total_triangles = 0

    for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
        filename, chunk_size=chunk_size
    ):
        if plane_in_aligned and rotation_matrix is not None and origin is not None:
            v1 = (v1 - origin) @ rotation_matrix
            v2 = (v2 - origin) @ rotation_matrix
            v3 = (v3 - origin) @ rotation_matrix

        z1 = v1[:, 2]
        z2 = v2[:, 2]
        z3 = v3[:, 2]

        z_min = np.minimum(np.minimum(z1, z2), z3)
        z_max = np.maximum(np.maximum(z1, z2), z3)
        mask = (z_max >= plane_z - eps) & (z_min <= plane_z + eps)
        if mask.any():
            chunk_segments = _slice_triangle_arrays(
                v1[mask], v2[mask], v3[mask], plane_z, eps, grid_step
            )
            segments.extend(chunk_segments)
            if len(segments) > max_segments:
                raise MemoryError(
                    f"Too many segments ({len(segments)}). Increase grid_step or use coarser slice."
                )

        if progress_callback:
            progress = (chunk_end / total_triangles) * 100
            progress_callback(progress, f"Slicing: {chunk_end:,}/{total_triangles:,}")

    return segments, total_triangles, len(segments)


def calculate_perimeter_aligned(filename, *, voxel_mm=0.5, grid_step=0.2,
                                delta=0.01, eps=1e-6, chunk_size=500000,
                                progress_callback=None):
    """Периметр отпечатка после выравнивания по OBB (плоскость низа)."""
    aligned = calculate_parallelepiped_volume_streaming(
        filename,
        voxel_mm=voxel_mm,
        chunk_size=chunk_size,
        pass_c=True,
    )
    if not aligned:
        return None

    rotation_matrix = np.asarray(aligned['rotation_matrix'], dtype=np.float64)
    origin = np.asarray(aligned['origin'], dtype=np.float64)
    min_z = float(aligned['min_coords'][2])

    segments, total_triangles, total_segments = slice_stl_plane_segments_binary(
        filename,
        z_plane=min_z,
        eps=eps,
        delta=delta,
        grid_step=grid_step,
        chunk_size=chunk_size,
        progress_callback=progress_callback,
        rotation_matrix=rotation_matrix,
        origin=origin,
        plane_in_aligned=True,
    )

    loops = _build_contours_from_segments(segments, grid_step=grid_step)
    perimeter = perimeter_from_loops(loops)
    return {
        'total_triangles': total_triangles,
        'segments_count': total_segments,
        'loops_count': len(loops),
        'perimeter': perimeter,
        'loops': loops,
        'grid_step': grid_step,
        'z_plane': min_z,
        'delta': delta,
        'rotation_matrix': rotation_matrix,
        'origin': origin,
    }


def _build_contours_from_segments(segments, grid_step):
    """Сборка замкнутых контуров из списка отрезков."""
    # Построение графа смежности
    adj = collections.defaultdict(list)
    for (x1, y1), (x2, y2) in segments:
        key1 = (x1, y1)
        key2 = (x2, y2)
        adj[key1].append(key2)
        adj[key2].append(key1)

    visited_edges = set()
    loops = []

    for start_key in list(adj.keys()):
        for neighbor in adj[start_key]:
            edge = (start_key, neighbor) if start_key < neighbor else (neighbor, start_key)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            path = [start_key, neighbor]
            prev = start_key
            curr = neighbor

            while True:
                next_key = None
                for nb in adj[curr]:
                    edge_nb = (curr, nb) if curr < nb else (nb, curr)
                    if edge_nb in visited_edges:
                        continue
                    next_key = nb
                    visited_edges.add(edge_nb)
                    break

                if next_key is None:
                    break

                path.append(next_key)
                prev, curr = curr, next_key

                if curr == start_key:
                    loops.append(path[:-1])
                    break

    loops_pts = []
    for loop in loops:
        coords = np.array([[p[0] * grid_step, p[1] * grid_step] for p in loop], dtype=np.float64)
        loops_pts.append(coords)
    return loops_pts


def perimeter_from_loops(loops, min_perimeter=1.0):
    """Суммарный периметр всех контуров."""
    total = 0.0
    for loop in loops:
        if len(loop) < 2:
            continue
        perim = 0.0
        prev = loop[-1]
        for curr in loop:
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            perim += math.hypot(dx, dy)
            prev = curr
        if perim >= min_perimeter:
            total += perim
    return total


def calculate_perimeter_z0(filename, z_plane=0.0, delta=0.01, grid_step=0.2,
                           eps=1e-6, chunk_size=500000, progress_callback=None):
    """Основная функция для вычисления периметра отпечатка на плоскости Z."""
    segments, total_triangles, total_segments = slice_stl_plane_segments_binary(
        filename, z_plane=z_plane, eps=eps, delta=delta,
        grid_step=grid_step, chunk_size=chunk_size,
        progress_callback=progress_callback
    )
    loops = _build_contours_from_segments(segments, grid_step=grid_step)
    perimeter = perimeter_from_loops(loops)
    return {
        'total_triangles': total_triangles,
        'segments_count': total_segments,
        'loops_count': len(loops),
        'perimeter': perimeter,
        'loops': loops,
        'grid_step': grid_step,
        'z_plane': z_plane,
        'delta': delta,
    }


def _run_slice_self_tests():
    """Самопроверки периметра на синтетических моделях."""
    def make_box(x0, y0, z0, x1, y1, z1):
        v = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], dtype=np.float64)
        tris = []
        # нижняя
        tris += [[v[0], v[1], v[2]], [v[0], v[2], v[3]]]
        # верхняя
        tris += [[v[4], v[6], v[5]], [v[4], v[7], v[6]]]
        # стороны
        tris += [[v[0], v[5], v[1]], [v[0], v[4], v[5]]]
        tris += [[v[1], v[6], v[2]], [v[1], v[5], v[6]]]
        tris += [[v[2], v[7], v[3]], [v[2], v[6], v[7]]]
        tris += [[v[3], v[4], v[0]], [v[3], v[7], v[4]]]
        return np.array(tris, dtype=np.float64)

    grid_step = 0.2
    delta = 0.01
    eps = 1e-6

    # Куб 10x10x10 на плоскости
    cube = make_box(0, 0, 0, 10, 10, 10)
    v1, v2, v3 = cube[:, 0], cube[:, 1], cube[:, 2]
    segments = _slice_triangle_arrays(v1, v2, v3, delta, eps, grid_step)
    loops = _build_contours_from_segments(segments, grid_step)
    perim = perimeter_from_loops(loops)
    assert abs(perim - 40.0) < 2.0, f"Cube perimeter mismatch: {perim}"

    # Рамка: внешний 20x20, внутренний 10x10
    outer = make_box(-10, -10, 0, 10, 10, 5)
    inner = make_box(-5, -5, 0, 5, 5, 5)
    ring = np.vstack([outer, inner])
    v1, v2, v3 = ring[:, 0], ring[:, 1], ring[:, 2]
    segments = _slice_triangle_arrays(v1, v2, v3, delta, eps, grid_step)
    loops = _build_contours_from_segments(segments, grid_step)
    perim = perimeter_from_loops(loops)
    assert abs(perim - (80.0 + 40.0)) < 4.0, f"Ring perimeter mismatch: {perim}"

    # Проверка с поворотом и выравниванием
    angle = np.deg2rad(30)
    rot_z = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)]
    ])
    rot = rot_z @ rot_x
    trans = np.array([15.0, -7.0, 3.0])
    cube_rot = (cube.reshape(-1, 3) @ rot.T + trans).reshape(cube.shape)

    # Имитируем alignment через расчет OBB
    v1, v2, v3 = cube_rot[:, 0], cube_rot[:, 1], cube_rot[:, 2]
    fake_vertices = np.stack([v1, v2, v3], axis=1)
    max_area, _, tri_vertices = find_max_area_triangle_vectorized(fake_vertices)
    edge1 = tri_vertices[1] - tri_vertices[0]
    edge2 = tri_vertices[2] - tri_vertices[0]
    normal = np.cross(edge1, edge2)
    min_c, max_c, R = calculate_aligned_bounding_box_optimized(
        fake_vertices, normal, tri_vertices[0]
    )
    aligned_min_z = float(min_c[2])

    segments = _slice_triangle_arrays(
        (v1 - tri_vertices[0]) @ R, (v2 - tri_vertices[0]) @ R, (v3 - tri_vertices[0]) @ R,
        aligned_min_z + delta, eps, grid_step
    )
    loops = _build_contours_from_segments(segments, grid_step)
    perim = perimeter_from_loops(loops)
    assert abs(perim - 40.0) < 3.0, f"Aligned rotated cube mismatch: {perim}"

    print("Slice self-tests passed.")


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


def _iter_stl_binary_chunks(filename, chunk_size=500000):
    """Итерация по чанкам бинарного STL без лишнего хранения в памяти."""
    with open(filename, 'rb') as f:
        f.read(80)
        num_triangles = struct.unpack('I', f.read(4))[0]

    mmap = np.memmap(filename, dtype=np.uint8, mode='r')
    data_offset = 84
    dtype = np.dtype([
        ('normal', '3f4'),
        ('v1', '3f4'),
        ('v2', '3f4'),
        ('v3', '3f4'),
        ('attr', 'u2')
    ])

    for chunk_start in range(0, num_triangles, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_triangles)
        chunk_byte_start = data_offset + chunk_start * 50
        chunk_byte_end = data_offset + chunk_end * 50
        chunk_data_slice = mmap[chunk_byte_start:chunk_byte_end]
        if len(chunk_data_slice) % dtype.itemsize != 0:
            raise ValueError(
                "Buffer size must be a multiple of element size. "
                "This indicates the file is likely an ASCII STL format or corrupted binary STL file. "
                "Please convert to binary STL or verify file integrity."
            )
        chunk_data = np.frombuffer(chunk_data_slice, dtype=dtype)
        yield chunk_start, chunk_end, num_triangles, chunk_data


def _iter_stl_ascii_chunks(filename, chunk_size=500000):
    """Итерация по чанкам ASCII STL, загружая весь файл и разбивая на чанки."""
    vertices, _ = _parse_stl_ascii_vectorized(filename)
    total_triangles = vertices.shape[0]

    if total_triangles == 0:
        return

    for chunk_start in range(0, total_triangles, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_triangles)
        v1 = vertices[chunk_start:chunk_end, 0].astype(np.float64)
        v2 = vertices[chunk_start:chunk_end, 1].astype(np.float64)
        v3 = vertices[chunk_start:chunk_end, 2].astype(np.float64)
        yield chunk_start, chunk_end, total_triangles, v1, v2, v3


def _iter_stl_chunks_auto(filename, chunk_size=500000):
    """Универсальный итератор по чанкам STL для бинарного и ASCII форматов."""
    if _is_binary_stl_fast(filename):
        for chunk_start, chunk_end, total_triangles, chunk_data in _iter_stl_binary_chunks(filename, chunk_size):
            v1 = chunk_data['v1'].astype(np.float64)
            v2 = chunk_data['v2'].astype(np.float64)
            v3 = chunk_data['v3'].astype(np.float64)
            yield chunk_start, chunk_end, total_triangles, v1, v2, v3
    else:
        for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_ascii_chunks(filename, chunk_size):
            yield chunk_start, chunk_end, total_triangles, v1, v2, v3



def calculate_parallelepiped_volume_streaming(
    filename,
    progress_callback=None,
    *,
    voxel_mm=0.5,
    chunk_size=500000,
    pass_c=True,
):
    """
    Экономная версия для больших STL файлов (>1M треугольников).
    Использует потоковый разбор с отображением в память и 2-3 прохода.
    """
    import numpy as np
    import time

    if progress_callback:
        progress_callback(0, "Starting STL processing...")

    start_time = time.time()

    # Pass A: поиск треугольника максимальной площади
    max_area = 0.0
    max_index = -1
    max_vertices = None
    max_normal = None

    total_triangles = 0
    for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
        filename, chunk_size=chunk_size
    ):
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        chunk_max_idx = np.argmax(areas)
        chunk_max_area = areas[chunk_max_idx]
        if chunk_max_area > max_area:
            max_area = float(chunk_max_area)
            max_index = chunk_start + int(chunk_max_idx)
            max_vertices = np.array([
                v1[chunk_max_idx],
                v2[chunk_max_idx],
                v3[chunk_max_idx]
            ], dtype=np.float64)
            max_normal = cross[chunk_max_idx]

        if progress_callback:
            progress = (chunk_end / total_triangles) * 30
            progress_callback(progress, f"Pass A: {chunk_end:,}/{total_triangles:,}")

    if max_vertices is None or total_triangles == 0:
        return None

    normal_norm = np.linalg.norm(max_normal)
    if normal_norm < 1e-12:
        return None

    z_axis = max_normal / normal_norm
    origin = max_vertices[0]

    # Базис в плоскости
    temp = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_base = np.cross(z_axis, temp)
    x_base = x_base / np.linalg.norm(x_base)
    y_base = np.cross(z_axis, x_base)
    y_base = y_base / np.linalg.norm(y_base)

    # Pass B: диапазон по Z и voxel thinning в плоскости
    z_min = np.inf
    z_max = -np.inf
    key_dtype = np.dtype([('x', 'i8'), ('y', 'i8')])
    voxel_keys = np.empty(0, dtype=key_dtype)
    key_limit = 5_000_000

    for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
        filename, chunk_size=chunk_size
    ):
        v_all = np.vstack([v1, v2, v3])

        v_trans = v_all - origin
        z_vals = v_trans @ z_axis
        z_min = min(z_min, float(z_vals.min()))
        z_max = max(z_max, float(z_vals.max()))

        x_vals = v_trans @ x_base
        y_vals = v_trans @ y_base

        qx = np.rint(x_vals / voxel_mm).astype(np.int64)
        qy = np.rint(y_vals / voxel_mm).astype(np.int64)

        keys_chunk = np.empty(len(qx), dtype=key_dtype)
        keys_chunk['x'] = qx
        keys_chunk['y'] = qy
        voxel_keys = np.concatenate([voxel_keys, keys_chunk])

        if len(voxel_keys) > key_limit:
            voxel_keys = np.unique(voxel_keys)

        if progress_callback:
            progress = 30 + (chunk_end / total_triangles) * 40
            progress_callback(progress, f"Pass B: {chunk_end:,}/{total_triangles:,}")

    if len(voxel_keys) == 0:
        return None

    voxel_keys = np.unique(voxel_keys)
    voxel_points = np.column_stack([
        voxel_keys['x'].astype(np.float64) * voxel_mm,
        voxel_keys['y'].astype(np.float64) * voxel_mm
    ])

    hull_points_2d = compute_convex_hull_2d(voxel_points)
    if len(hull_points_2d) == 0:
        return None

    # Поиск минимального прямоугольника (O(h))
    angle, width, height, min_area = rotating_calipers_min_area_rectangle(hull_points_2d)

    x_axis = np.cos(angle) * x_base + np.sin(angle) * y_base
    y_axis = -np.sin(angle) * x_base + np.cos(angle) * y_base
    best_rotation = np.column_stack([x_axis, y_axis, z_axis])

    # Pass C: точные экстенты в финальной системе
    min_coords = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    max_coords = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    if pass_c:
        for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
            filename, chunk_size=chunk_size
        ):
            v_all = np.vstack([v1, v2, v3])
            transformed = (v_all - origin) @ best_rotation
            min_coords = np.minimum(min_coords, transformed.min(axis=0))
            max_coords = np.maximum(max_coords, transformed.max(axis=0))

            if progress_callback:
                progress = 70 + (chunk_end / total_triangles) * 30
                progress_callback(progress, f"Pass C: {chunk_end:,}/{total_triangles:,}")
    else:
        min_coords = np.array([-width / 2, -height / 2, z_min], dtype=np.float64)
        max_coords = np.array([width / 2, height / 2, z_max], dtype=np.float64)

    dimensions = max_coords - min_coords
    volume = dimensions[0] * dimensions[1] * dimensions[2]

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
        'rotation_matrix': best_rotation,
        'origin': origin.tolist(),
        'voxel_mm': voxel_mm,
        'min_area_2d': min_area,
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

    # Проекция вершин на плоскость, перпендикулярную нормали (в базисе плоскости)
    v_translated = (unique_vertices - origin).astype(np.float64)
    projection_lengths = np.dot(v_translated, z_axis)
    projection_lengths_3d = projection_lengths[:, np.newaxis]
    projected_3d = v_translated - projection_lengths_3d * z_axis

    temp = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_base = np.cross(z_axis, temp)
    x_base = x_base / np.linalg.norm(x_base)
    y_base = np.cross(z_axis, x_base)
    y_base = y_base / np.linalg.norm(y_base)

    x_coords = np.dot(projected_3d, x_base)
    y_coords = np.dot(projected_3d, y_base)
    projected_2d = np.column_stack([x_coords, y_coords])

    hull_points_2d = compute_convex_hull_2d(projected_2d)

    if len(hull_points_2d) == 0:
        return np.zeros(3), np.zeros(3), np.eye(3)

    # Поиск минимального прямоугольника с использованием вращающихся штангенциркулей
    angle, width, height, min_area = rotating_calipers_min_area_rectangle(hull_points_2d)

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


def get_bounding_box_dimensions_streaming(filename, progress_callback=None):
    """Потоковая версия get_bounding_box_dimensions."""
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


def _remove_collinear_ccw(points, tol=1e-12):
    """Удаление коллинеарных соседних вершин из CCW выпуклого многоугольника."""
    n = len(points)
    if n < 3:
        return points

    cleaned = []
    for i in range(n):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[(i + 1) % n]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) > tol:
            cleaned.append(p_curr)

    return np.array(cleaned, dtype=np.float64)


def rotating_calipers_min_area_rectangle(points):
    """
    Минимальный прямоугольник для CCW выпуклого многоугольника за O(h).
    Возвращает: (angle, width, height, min_area)
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0, 0.0, 0.0, 0.0

    pts = _remove_collinear_ccw(pts)
    n = len(pts)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0

    edges = np.roll(pts, -1, axis=0) - pts

    def _advance_max(idx, direction):
        best = idx
        best_proj = float(np.dot(pts[best], direction))
        while True:
            nxt = (best + 1) % n
            nxt_proj = float(np.dot(pts[nxt], direction))
            if nxt_proj > best_proj + 1e-12:
                best = nxt
                best_proj = nxt_proj
            else:
                break
        return best, best_proj

    def _advance_min(idx, direction):
        best = idx
        best_proj = float(np.dot(pts[best], direction))
        while True:
            nxt = (best + 1) % n
            nxt_proj = float(np.dot(pts[nxt], direction))
            if nxt_proj < best_proj - 1e-12:
                best = nxt
                best_proj = nxt_proj
            else:
                break
        return best, best_proj

    # Инициализация экстремумов
    u0 = edges[0]
    u0 = u0 / np.linalg.norm(u0)
    v0 = np.array([-u0[1], u0[0]])

    proj_u = pts @ u0
    proj_v = pts @ v0
    i_max_u = int(np.argmax(proj_u))
    i_min_u = int(np.argmin(proj_u))
    i_max_v = int(np.argmax(proj_v))
    i_min_v = int(np.argmin(proj_v))

    min_area = float('inf')
    best_angle = 0.0
    best_width = 0.0
    best_height = 0.0

    for i in range(n):
        edge = edges[i]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        u = edge / edge_len
        v = np.array([-u[1], u[0]])

        i_max_u, max_u = _advance_max(i_max_u, u)
        i_min_u, min_u = _advance_min(i_min_u, u)
        i_max_v, max_v = _advance_max(i_max_v, v)
        i_min_v, min_v = _advance_min(i_min_v, v)

        width = max_u - min_u
        height = max_v - min_v
        area = width * height

        if area < min_area:
            min_area = area
            best_angle = float(np.arctan2(u[1], u[0]))
            best_width = float(width)
            best_height = float(height)

    return best_angle, best_width, best_height, min_area


def _bruteforce_min_area_rectangle(points, step_deg=1.0):
    """Грубая проверка через перебор углов."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 0.0, 0.0, 0.0, 0.0
    min_area = float('inf')
    best_angle = 0.0
    best_width = 0.0
    best_height = 0.0
    for angle_deg in np.arange(0.0, 180.0, step_deg):
        angle = np.deg2rad(angle_deg)
        u = np.array([np.cos(angle), np.sin(angle)])
        v = np.array([-u[1], u[0]])
        proj_u = pts @ u
        proj_v = pts @ v
        width = proj_u.max() - proj_u.min()
        height = proj_v.max() - proj_v.min()
        area = width * height
        if area < min_area:
            min_area = area
            best_angle = angle
            best_width = width
            best_height = height
    return best_angle, best_width, best_height, min_area


def _run_self_checks():
    """Минимальные самопроверки вращающихся калиперов без внешних файлов."""
    np.random.seed(0)
    voxel_mm = 0.5

    def make_rotated_rectangle(width, height, angle_deg):
        hw, hh = width / 2, height / 2
        rect = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ], dtype=np.float64)
        angle = np.deg2rad(angle_deg)
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return rect @ rot.T

    cases = [
        (1000.0, 1500.0, 45.0),
        (800.0, 500.0, 33.0),
        (1200.0, 200.0, 12.0)
    ]

    for width, height, angle in cases:
        corners = make_rotated_rectangle(width, height, angle)
        jitter = np.random.uniform(-0.2, 0.2, size=(200, 2))
        samples = np.vstack([corners, corners + jitter])

        qx = np.rint(samples[:, 0] / voxel_mm).astype(np.int64)
        qy = np.rint(samples[:, 1] / voxel_mm).astype(np.int64)
        pts = np.column_stack([qx * voxel_mm, qy * voxel_mm])

        hull_pts = compute_convex_hull_2d(pts)
        rc_angle, rc_w, rc_h, rc_area = rotating_calipers_min_area_rectangle(hull_pts)
        bf_angle, bf_w, bf_h, bf_area = _bruteforce_min_area_rectangle(hull_pts, step_deg=1.0)

        assert rc_area <= bf_area + 1e-6, "Rotating calipers хуже brute-force"
        assert abs(rc_w - width) <= 1.0 or abs(rc_h - width) <= 1.0, "Ширина >1мм"
        assert abs(rc_h - height) <= 1.0 or abs(rc_w - height) <= 1.0, "Высота >1мм"

    print("Self-checks passed.")


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

    # Уникальные вершины без округления координат
    unique_vertices = np.unique(all_vertices, axis=0)

    # Проекция вершин на плоскость, перпендикулярную нормали
    v_translated = (unique_vertices - origin).astype(np.float64)
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
    angle, width, height, min_area = rotating_calipers_min_area_rectangle(hull_points_2d)

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
    # dimensions_sorted = dimensions[sorted_indices]
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
            unique_vertices.add(tuple(vertex))

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





def get_bounding_box_dimensions(filename):
    """Вернуть размеры габаритного бокса (X, Y, Z) для STL."""
    result = calculate_parallelepiped_volume(filename)

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
        print("  --perimeter-z0    Рассчитать периметр отпечатка на плоскости Z=0")
        print("\nПримеры:")
        print("  python stl.py model.stl")
        print("  python stl.py model.stl --volume")
        print("  python stl.py model.stl --perimeter-z0")
        return

    filename = sys.argv[1]
    calculate_volume = "--volume" in sys.argv
    calculate_perimeter = "--perimeter" in sys.argv

    try:
        if calculate_volume:
            result = calculate_parallelepiped_volume_streaming(filename)

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
        elif calculate_perimeter:
            result = calculate_perimeter_aligned(filename)

            if result:
                print("\n" + "=" * 60)
                print("Периметр отпечатка на плоскости")
                print("=" * 60)
                print(f"Плоскость Z: {result['z_plane']:.3f} (delta={result['delta']:.3f})")
                print(f"Сегментов: {result['segments_count']}")
                print(f"Контуров: {result['loops_count']}")
                print(f"\n  ПЕРИМЕТР: {result['perimeter']:.6f}")
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
