import struct
import numpy as np
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import collections
import math



try:
    from numba import njit
    _NUMBA_AVAILABLE = True 
except Exception:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _rasterize_triangles_numba(grid, v1_xy, v2_xy, v3_xy, min_x, min_y, cell_mm):
        rows, cols = grid.shape
        for i in range(v1_xy.shape[0]):
            x1, y1 = v1_xy[i, 0], v1_xy[i, 1]
            x2, y2 = v2_xy[i, 0], v2_xy[i, 1]
            x3, y3 = v3_xy[i, 0], v3_xy[i, 1]

            minx = x1 if x1 < x2 else x2
            minx = minx if minx < x3 else x3
            maxx = x1 if x1 > x2 else x2
            maxx = maxx if maxx > x3 else x3
            miny = y1 if y1 < y2 else y2
            miny = miny if miny < y3 else y3
            maxy = y1 if y1 > y2 else y2
            maxy = maxy if maxy > y3 else y3

            ix0 = int(np.floor((minx - min_x) / cell_mm))
            ix1 = int(np.floor((maxx - min_x) / cell_mm))
            iy0 = int(np.floor((miny - min_y) / cell_mm))
            iy1 = int(np.floor((maxy - min_y) / cell_mm))

            if ix1 < 0 or iy1 < 0 or ix0 >= cols or iy0 >= rows:
                continue
            if ix0 < 0:
                ix0 = 0
            if iy0 < 0:
                iy0 = 0
            if ix1 >= cols:
                ix1 = cols - 1
            if iy1 >= rows:
                iy1 = rows - 1

            for yy in range(iy0, iy1 + 1):
                py = min_y + (yy + 0.5) * cell_mm
                for xx in range(ix0, ix1 + 1):
                    px = min_x + (xx + 0.5) * cell_mm
                    d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
                    d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3)
                    d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1)
                    has_neg = (d1 < 0.0) or (d2 < 0.0) or (d3 < 0.0)
                    has_pos = (d1 > 0.0) or (d2 > 0.0) or (d3 > 0.0)
                    if not (has_neg and has_pos):
                        grid[yy, xx] = True

    @njit(cache=True)
    def _rasterize_triangles_zmax_numba(occ, zmax, v1, v2, v3, min_x, min_y, cell_mm):
        rows, cols = occ.shape
        for i in range(v1.shape[0]):
            x1, y1, z1 = v1[i, 0], v1[i, 1], v1[i, 2]
            x2, y2, z2 = v2[i, 0], v2[i, 1], v2[i, 2]
            x3, y3, z3 = v3[i, 0], v3[i, 1], v3[i, 2]

            minx = x1 if x1 < x2 else x2
            minx = minx if minx < x3 else x3
            maxx = x1 if x1 > x2 else x2
            maxx = maxx if maxx > x3 else x3
            miny = y1 if y1 < y2 else y2
            miny = miny if miny < y3 else y3
            maxy = y1 if y1 > y2 else y2
            maxy = maxy if maxy > y3 else y3

            ix0 = int(np.floor((minx - min_x) / cell_mm))
            ix1 = int(np.floor((maxx - min_x) / cell_mm))
            iy0 = int(np.floor((miny - min_y) / cell_mm))
            iy1 = int(np.floor((maxy - min_y) / cell_mm))

            if ix1 < 0 or iy1 < 0 or ix0 >= cols or iy0 >= rows:
                continue
            if ix0 < 0:
                ix0 = 0
            if iy0 < 0:
                iy0 = 0
            if ix1 >= cols:
                ix1 = cols - 1
            if iy1 >= rows:
                iy1 = rows - 1

            denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if np.abs(denom) < 1e-12:
                continue

            for yy in range(iy0, iy1 + 1):
                py = min_y + (yy + 0.5) * cell_mm
                for xx in range(ix0, ix1 + 1):
                    px = min_x + (xx + 0.5) * cell_mm
                    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
                    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
                    w3 = 1.0 - w1 - w2
                    if w1 >= -1e-12 and w2 >= -1e-12 and w3 >= -1e-12:
                        z = w1 * z1 + w2 * z2 + w3 * z3
                        occ[yy, xx] = True
                        if z > zmax[yy, xx]:
                            zmax[yy, xx] = z


def _rasterize_triangles_numpy(grid, v1_xy, v2_xy, v3_xy, min_x, min_y, cell_mm):
    rows, cols = grid.shape
    for i in range(v1_xy.shape[0]):
        x1, y1 = v1_xy[i]
        x2, y2 = v2_xy[i]
        x3, y3 = v3_xy[i]

        minx = min(x1, x2, x3)
        maxx = max(x1, x2, x3)
        miny = min(y1, y2, y3)
        maxy = max(y1, y2, y3)

        ix0 = int(np.floor((minx - min_x) / cell_mm))
        ix1 = int(np.floor((maxx - min_x) / cell_mm))
        iy0 = int(np.floor((miny - min_y) / cell_mm))
        iy1 = int(np.floor((maxy - min_y) / cell_mm))

        if ix1 < 0 or iy1 < 0 or ix0 >= cols or iy0 >= rows:
            continue
        ix0 = max(ix0, 0)
        iy0 = max(iy0, 0)
        ix1 = min(ix1, cols - 1)
        iy1 = min(iy1, rows - 1)

        xs = np.arange(ix0, ix1 + 1)
        ys = np.arange(iy0, iy1 + 1)
        px = min_x + (xs + 0.5) * cell_mm
        py = min_y + (ys + 0.5) * cell_mm
        px_grid, py_grid = np.meshgrid(px, py)

        d1 = (px_grid - x2) * (y1 - y2) - (x1 - x2) * (py_grid - y2)
        d2 = (px_grid - x3) * (y2 - y3) - (x2 - x3) * (py_grid - y3)
        d3 = (px_grid - x1) * (y3 - y1) - (x3 - x1) * (py_grid - y1)
        mask = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))

        grid[np.ix_(ys, xs)] |= mask


def _rasterize_triangles_zmax_numpy(occ, zmax, v1, v2, v3, min_x, min_y, cell_mm):
    rows, cols = occ.shape
    for i in range(v1.shape[0]):
        x1, y1, z1 = v1[i]
        x2, y2, z2 = v2[i]
        x3, y3, z3 = v3[i]

        minx = min(x1, x2, x3)
        maxx = max(x1, x2, x3)
        miny = min(y1, y2, y3)
        maxy = max(y1, y2, y3)

        ix0 = int(np.floor((minx - min_x) / cell_mm))
        ix1 = int(np.floor((maxx - min_x) / cell_mm))
        iy0 = int(np.floor((miny - min_y) / cell_mm))
        iy1 = int(np.floor((maxy - min_y) / cell_mm))

        if ix1 < 0 or iy1 < 0 or ix0 >= cols or iy0 >= rows:
            continue
        ix0 = max(ix0, 0)
        iy0 = max(iy0, 0)
        ix1 = min(ix1, cols - 1)
        iy1 = min(iy1, rows - 1)

        xs = np.arange(ix0, ix1 + 1)
        ys = np.arange(iy0, iy1 + 1)
        px = min_x + (xs + 0.5) * cell_mm
        py = min_y + (ys + 0.5) * cell_mm
        px_grid, py_grid = np.meshgrid(px, py)

        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-12:
            continue

        w1 = ((y2 - y3) * (px_grid - x3) + (x3 - x2) * (py_grid - y3)) / denom
        w2 = ((y3 - y1) * (px_grid - x3) + (x1 - x3) * (py_grid - y3)) / denom
        w3 = 1.0 - w1 - w2
        mask = (w1 >= -1e-12) & (w2 >= -1e-12) & (w3 >= -1e-12)
        if not np.any(mask):
            continue

        z = w1 * z1 + w2 * z2 + w3 * z3
        zmax_block = zmax[np.ix_(ys, xs)]
        zmax_block = np.where(mask, np.maximum(zmax_block, z), zmax_block)
        zmax[np.ix_(ys, xs)] = zmax_block
        occ[np.ix_(ys, xs)] |= mask


def calculate_contact_perimeter_projected(
    filename,
    *,
    cell_mm=1.0,
    chunk_size=500000,
    max_cells=10_000_000,
    progress_callback=None,
):
    """
    Быстрый оценочный периметр контакта через растровую сетку.
    Возвращает dict: perimeter_mm, cell_mm, total_triangles, projected_triangles.
    """
    from skimage.measure import perimeter_crofton
    import tempfile

    max_area = 0.0
    max_vertices = None
    total_triangles = 0

    for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
        filename, chunk_size=chunk_size
    ):
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        chunk_max_idx = np.argmax(areas)
        chunk_max_area = float(areas[chunk_max_idx])
        if chunk_max_area > max_area:
            max_area = chunk_max_area
            max_vertices = np.array([
                v1[chunk_max_idx],
                v2[chunk_max_idx],
                v3[chunk_max_idx],
            ], dtype=np.float64)

        if progress_callback:
            progress = (chunk_end / total_triangles) * 10
            progress_callback(progress, f"Max triangle: {chunk_end:,}/{total_triangles:,}")

    if max_vertices is None:
        return None

    rotation_matrix, origin = create_coordinate_system_from_triangle_vectorized(max_vertices)

    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf

    for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
        filename, chunk_size=chunk_size
    ):
        v1_local = (v1 - origin) @ rotation_matrix
        v2_local = (v2 - origin) @ rotation_matrix
        v3_local = (v3 - origin) @ rotation_matrix

        v1_xy = v1_local[:, :2]
        v2_xy = v2_local[:, :2]
        v3_xy = v3_local[:, :2]

        min_x = min(min_x, v1_xy[:, 0].min(), v2_xy[:, 0].min(), v3_xy[:, 0].min())
        min_y = min(min_y, v1_xy[:, 1].min(), v2_xy[:, 1].min(), v3_xy[:, 1].min())
        max_x = max(max_x, v1_xy[:, 0].max(), v2_xy[:, 0].max(), v3_xy[:, 0].max())
        max_y = max(max_y, v1_xy[:, 1].max(), v2_xy[:, 1].max(), v3_xy[:, 1].max())

        if progress_callback:
            progress = 10 + (chunk_end / total_triangles) * 20
            progress_callback(progress, f"Bounds: {chunk_end:,}/{total_triangles:,}")

    if not np.isfinite(min_x) or not np.isfinite(min_y):
        return None

    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return {
            'perimeter_mm': 0.0,
            'cell_mm': cell_mm,
            'total_triangles': total_triangles,
            'projected_triangles': 0,
        }

    cols = int(np.ceil(width / cell_mm)) + 2
    rows = int(np.ceil(height / cell_mm)) + 2
    total_cells = rows * cols

    if total_cells > max_cells:
        temp_file = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        temp_file.close()
        grid = np.memmap(temp_file.name, dtype=np.bool_, mode="w+", shape=(rows, cols))
    else:
        grid = np.zeros((rows, cols), dtype=np.bool_)

    grid.fill(False)

    projected_triangles = 0
    for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
        filename, chunk_size=chunk_size
    ):
        v1_local = (v1 - origin) @ rotation_matrix
        v2_local = (v2 - origin) @ rotation_matrix
        v3_local = (v3 - origin) @ rotation_matrix

        v1_xy = v1_local[:, :2]
        v2_xy = v2_local[:, :2]
        v3_xy = v3_local[:, :2]

        area2 = np.abs(
            (v2_xy[:, 0] - v1_xy[:, 0]) * (v3_xy[:, 1] - v1_xy[:, 1])
            - (v2_xy[:, 1] - v1_xy[:, 1]) * (v3_xy[:, 0] - v1_xy[:, 0])
        )
        valid = area2 > 1e-5
        if np.any(valid):
            projected_triangles += int(np.sum(valid))
            v1_valid = v1_xy[valid]
            v2_valid = v2_xy[valid]
            v3_valid = v3_xy[valid]
            if _NUMBA_AVAILABLE:
                _rasterize_triangles_numba(grid, v1_valid, v2_valid, v3_valid, min_x, min_y, cell_mm)
            else:
                _rasterize_triangles_numpy(grid, v1_valid, v2_valid, v3_valid, min_x, min_y, cell_mm)

        if progress_callback:
            progress = 30 + (chunk_end / total_triangles) * 70
            progress_callback(progress, f"Raster: {chunk_end:,}/{total_triangles:,}")

    perimeter_pixels = float(perimeter_crofton(grid.astype(np.uint8), directions=4))
    perimeter_mm = perimeter_pixels * cell_mm

    return {
        'perimeter_mm': perimeter_mm,
        'cell_mm': cell_mm,
        'total_triangles': total_triangles,
        'projected_triangles': projected_triangles,
    }


def calculate_removed_volume_from_top(
    filename,
    *,
    cell_mm=10.0,
    chunk_size=500000,
    max_cells=10_000_000,
    progress_callback=None,
):
    """
    Оценка объёма снятого материала сверху по растровой сетке.
    Возвращает dict: removed_volume_mm3, cell_mm, total_triangles, projected_triangles, z_stock.
    """
    import tempfile

    max_area = 0.0
    max_vertices = None
    total_triangles = 0

    for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
        filename, chunk_size=chunk_size
    ):
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        chunk_max_idx = np.argmax(areas)
        chunk_max_area = float(areas[chunk_max_idx])
        if chunk_max_area > max_area:
            max_area = chunk_max_area
            max_vertices = np.array([
                v1[chunk_max_idx],
                v2[chunk_max_idx],
                v3[chunk_max_idx],
            ], dtype=np.float64)

        if progress_callback:
            progress = (chunk_end / total_triangles) * 10
            progress_callback(progress, f"Max triangle: {chunk_end:,}/{total_triangles:,}")

    if max_vertices is None:
        return None

    rotation_matrix, origin = create_coordinate_system_from_triangle_vectorized(max_vertices)

    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    z_stock = -np.inf

    for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
        filename, chunk_size=chunk_size
    ):
        v1_local = (v1 - origin) @ rotation_matrix
        v2_local = (v2 - origin) @ rotation_matrix
        v3_local = (v3 - origin) @ rotation_matrix

        v1_xy = v1_local[:, :2]
        v2_xy = v2_local[:, :2]
        v3_xy = v3_local[:, :2]

        min_x = min(min_x, v1_xy[:, 0].min(), v2_xy[:, 0].min(), v3_xy[:, 0].min())
        min_y = min(min_y, v1_xy[:, 1].min(), v2_xy[:, 1].min(), v3_xy[:, 1].min())
        max_x = max(max_x, v1_xy[:, 0].max(), v2_xy[:, 0].max(), v3_xy[:, 0].max())
        max_y = max(max_y, v1_xy[:, 1].max(), v2_xy[:, 1].max(), v3_xy[:, 1].max())
        z_stock = max(z_stock, v1_local[:, 2].max(), v2_local[:, 2].max(), v3_local[:, 2].max())

        if progress_callback:
            progress = 10 + (chunk_end / total_triangles) * 20
            progress_callback(progress, f"Bounds: {chunk_end:,}/{total_triangles:,}")

    if not np.isfinite(min_x) or not np.isfinite(min_y) or not np.isfinite(z_stock):
        return None

    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return {
            'removed_volume_mm3': 0.0,
            'cell_mm': cell_mm,
            'total_triangles': total_triangles,
            'projected_triangles': 0,
            'z_stock': float(z_stock),
        }

    cols = int(np.ceil(width / cell_mm)) + 2
    rows = int(np.ceil(height / cell_mm)) + 2
    total_cells = rows * cols

    if total_cells > max_cells:
        temp_occ = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        temp_occ.close()
        temp_footprint = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        temp_footprint.close()
        temp_zmax = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        temp_zmax.close()
        occ = np.memmap(temp_occ.name, dtype=np.bool_, mode="w+", shape=(rows, cols))
        footprint = np.memmap(temp_footprint.name, dtype=np.bool_, mode="w+", shape=(rows, cols))
        zmax = np.memmap(temp_zmax.name, dtype=np.float32, mode="w+", shape=(rows, cols))
    else:
        occ = np.zeros((rows, cols), dtype=np.bool_)
        footprint = np.zeros((rows, cols), dtype=np.bool_)
        zmax = np.zeros((rows, cols), dtype=np.float32)

    occ.fill(False)
    footprint.fill(False)
    zmax.fill(-np.inf)

    projected_triangles = 0
    for chunk_start, chunk_end, total_triangles, v1, v2, v3 in _iter_stl_chunks_auto(
        filename, chunk_size=chunk_size
    ):
        v1_local = (v1 - origin) @ rotation_matrix
        v2_local = (v2 - origin) @ rotation_matrix
        v3_local = (v3 - origin) @ rotation_matrix

        v1_xy = v1_local[:, :2]
        v2_xy = v2_local[:, :2]
        v3_xy = v3_local[:, :2]

        area2 = np.abs(
            (v2_xy[:, 0] - v1_xy[:, 0]) * (v3_xy[:, 1] - v1_xy[:, 1])
            - (v2_xy[:, 1] - v1_xy[:, 1]) * (v3_xy[:, 0] - v1_xy[:, 0])
        )
        valid = area2 > 1e-5
        if np.any(valid):
            projected_triangles += int(np.sum(valid))
            v1_xy_valid = v1_xy[valid]
            v2_xy_valid = v2_xy[valid]
            v3_xy_valid = v3_xy[valid]
            v1_valid = v1_local[valid]
            v2_valid = v2_local[valid]
            v3_valid = v3_local[valid]

            if _NUMBA_AVAILABLE:
                _rasterize_triangles_zmax_numba(occ, zmax, v1_valid, v2_valid, v3_valid, min_x, min_y, cell_mm)
                _rasterize_triangles_numba(footprint, v1_xy_valid, v2_xy_valid, v3_xy_valid, min_x, min_y, cell_mm)
            else:
                _rasterize_triangles_zmax_numpy(occ, zmax, v1_valid, v2_valid, v3_valid, min_x, min_y, cell_mm)
                _rasterize_triangles_numpy(footprint, v1_xy_valid, v2_xy_valid, v3_xy_valid, min_x, min_y, cell_mm)

        if progress_callback:
            progress = 30 + (chunk_end / total_triangles) * 70
            progress_callback(progress, f"Raster: {chunk_end:,}/{total_triangles:,}")

    valid_mask = footprint & np.isfinite(zmax)
    if not np.any(valid_mask):
        removed_volume_mm3 = 0.0
    else:
        removed_volume_mm3 = float(np.sum((z_stock - zmax)[valid_mask]) * (cell_mm ** 2))

    return {
        'removed_volume_mm3': removed_volume_mm3,
        'cell_mm': cell_mm,
        'total_triangles': total_triangles,
        'projected_triangles': projected_triangles,
        'z_stock': float(z_stock),
    }


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

    # Ось X: проекция более длинного ребра на плоскость (перпендикуляр к Z)
    if np.dot(edge2, edge2) > np.dot(edge1, edge1):
        edge_base = edge2
    else:
        edge_base = edge1

    x_axis = edge_base - np.dot(edge_base, z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-10:
        # Ребро слишком короткое, используем произвольный перпендикулярный вектор
        if abs(z_axis[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, temp)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-12:
            return np.eye(3), v1

    x_axis = x_axis / x_norm

    # Ось Y: перпендикулярна X и Z (всегда нормализуем)
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-12:
        return np.eye(3), v1
    y_axis = y_axis / y_norm

    # Перепроверка ортонормированности (устраняем накопленную погрешность)
    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-12:
        return np.eye(3), v1
    x_axis = x_axis / x_norm

    # Матрица вращения (столбцы - базисные векторы)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    return rotation_matrix, v1


def _convex_hull_monotone_chain(points):
    """
    Вычисление выпуклой оболочки 2D точек по алгоритму Эндрю (monotone chain).
    Возвращает вершины оболочки в порядке против часовой стрелки.
    Если все точки коллинеарны, возвращает крайние точки.
    """
    pts = np.asarray(points)
    if pts.shape[0] < 3:
        return pts
    
    # Сортируем по x, затем по y
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts_sorted = pts[order]
    
    # Удаляем дубликаты (точки с одинаковыми координатами)
    # Используем разность по модулю, чтобы учесть численные погрешности
    eps = 1e-12
    diff = np.diff(pts_sorted, axis=0)
    keep = np.any(np.abs(diff) > eps, axis=1)
    # Первая точка всегда включается
    keep = np.concatenate([[True], keep])
    pts_unique = pts_sorted[keep]
    
    if pts_unique.shape[0] < 3:
        return pts_unique
    
    # Нижняя оболочка
    lower = []
    for p in pts_unique:
        while len(lower) >= 2:
            v1 = lower[-1] - lower[-2]
            v2 = p - lower[-2]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross <= eps:
                lower.pop()
            else:
                break
        lower.append(p)
    
    # Верхняя оболочка
    upper = []
    for p in reversed(pts_unique):
        while len(upper) >= 2:
            v1 = upper[-1] - upper[-2]
            v2 = p - upper[-2]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross <= eps:
                upper.pop()
            else:
                break
        upper.append(p)
    
    # Объединяем нижнюю и верхнюю оболочки, удаляя последнюю точку верхней оболочки,
    # так как она совпадает с первой точкой нижней.
    hull = np.vstack([lower[:-1], upper[:-1]])
    return hull


def compute_convex_hull_2d(points):
    """
    Вычисление 2D выпуклой оболочки точек с использованием SciPy.
    points: (n, 2) массив 2D точек
    Возвращает: hull_points (m, 2) массив вершин оболочки в порядке против часовой стрелки
    """
    if len(points) < 3:
        return points
    
    pts = np.asarray(points, dtype=np.float64)
    
    # 1. Удаление дубликатов с учётом квантования voxel_mm
    # Поскольку точки уже квантованы (voxel_keys), просто используем уникальные координаты
    pts_unique = np.unique(pts, axis=0)
    
    # 2. Ограничение размера выборки для предотвращения ошибок Qhull
    MAX_HULL_POINTS = 500_000
    if pts_unique.shape[0] > MAX_HULL_POINTS:
        # Случайная выборка с сохранением равномерного распределения
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(pts_unique.shape[0], size=MAX_HULL_POINTS, replace=False)
        pts_unique = pts_unique[indices]
    
    # 3. Попытка SciPy ConvexHull с опцией "QJ" (joggle) для улучшения устойчивости
    try:
        hull = ConvexHull(pts_unique, qhull_options="QJ")
        hull_points = pts_unique[hull.vertices]
        # Проверяем направление обхода (должно быть против часовой стрелки)
        # Если площадь отрицательна, меняем порядок
        if len(hull_points) >= 3:
            # Вычисляем знак площади (формула шнурования)
            total = 0.0
            for i in range(len(hull_points)):
                x1, y1 = hull_points[i]
                x2, y2 = hull_points[(i + 1) % len(hull_points)]
                total += (x2 - x1) * (y2 + y1)
            if total < 0:  # по часовой стрелке
                hull_points = hull_points[::-1]
        return hull_points
    except Exception:
        # 4. Резервный вариант: monotone chain
        hull_points = _convex_hull_monotone_chain(pts_unique)
        return hull_points


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
        print("\nПримеры:")
        print("  python stl.py model.stl")
        print("  python stl.py model.stl --volume")
        return

    filename = sys.argv[1]
    calculate_volume = "--volume" in sys.argv

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
