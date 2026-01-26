import struct
import numpy as np
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ========== Optimized versions ==========

def _is_binary_stl_fast(filename):
    """Fast binary STL detection."""
    import os
    file_size = os.path.getsize(filename)
    
    # Quick check: if file size < 84 bytes, can't be binary
    if file_size < 84:
        return False
    
    with open(filename, 'rb') as f:
        header = f.read(80)
        # Check if header contains non-ASCII characters
        try:
            header.decode('ascii')
            # If decodes successfully, check for 'solid' at beginning
            if header[:5].lower() == b'solid':
                # Might still be binary if size matches binary format
                pass
        except UnicodeDecodeError:
            return True
        
        # Check binary structure
        f.seek(80)
        num_triangles_bytes = f.read(4)
        if len(num_triangles_bytes) < 4:
            return False
        num_triangles = struct.unpack('I', num_triangles_bytes)[0]
        expected_size = 80 + 4 + num_triangles * 50
        return file_size == expected_size


def _parse_stl_binary_vectorized(filename):
    """Parse binary STL with vectorized output."""
    with open(filename, 'rb') as f:
        f.read(80)  # Skip header
        num_triangles = struct.unpack('I', f.read(4))[0]
        
        # Pre-allocate arrays
        vertices = np.zeros((num_triangles, 3, 3), dtype=np.float32)
        normals = np.zeros((num_triangles, 3), dtype=np.float32)
        
        for i in range(num_triangles):
            # Read normal (3 floats)
            normal = struct.unpack('fff', f.read(12))
            normals[i] = normal
            
            # Read vertices (9 floats)
            v1 = struct.unpack('fff', f.read(12))
            v2 = struct.unpack('fff', f.read(12))
            v3 = struct.unpack('fff', f.read(12))
            vertices[i, 0] = v1
            vertices[i, 1] = v2
            vertices[i, 2] = v3
            
            # Skip attribute bytes
            f.read(2)
        
        # Create compatibility dicts
        triangles = []
        for i in range(num_triangles):
            triangles.append({
                'normal': normals[i].tolist(),
                'vertices': vertices[i].tolist()
            })
        
        return vertices, triangles


def _parse_stl_ascii_vectorized(filename):
    """Parse ASCII STL with vectorized output."""
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
    Parse STL file and return vertices as numpy arrays.
    Returns: (vertices, triangles) where:
        vertices: (n, 3, 3) array of triangle vertices
        triangles: list of original dicts for compatibility
    """
    import time
    start_time = time.time()

    if _is_binary_stl_fast(filename):
        tmp = _parse_stl_binary_vectorized(filename)
        print("--- %s seconds ---" % (time.time() - start_time))
        return tmp
        # return _parse_stl_binary_vectorized(filename)
    else:
        tmp = _parse_stl_ascii_vectorized(filename)
        print("--- %s seconds ---" % (time.time() - start_time))
        return tmp
        # return _parse_stl_ascii_vectorized(filename)


def calculate_triangle_areas_vectorized(vertices):
    """
    Calculate triangle areas using vectorized operations.
    vertices: (n, 3, 3) array where vertices[i, j] is j-th vertex of i-th triangle
    Returns: (n,) array of areas
    """
    # Edge vectors
    v1 = vertices[:, 0]  # (n, 3)
    v2 = vertices[:, 1]  # (n, 3)
    v3 = vertices[:, 2]  # (n, 3)
    
    edge1 = v2 - v1  # (n, 3)
    edge2 = v3 - v1  # (n, 3)
    
    # Cross product
    cross = np.cross(edge1, edge2)  # (n, 3)
    
    # Area = 0.5 * norm of cross product
    areas = 0.5 * np.linalg.norm(cross, axis=1)  # (n,)
    
    return areas


def find_max_area_triangle_vectorized(vertices):
    """
    Find triangle with maximum area.
    Returns: (max_area, max_index, max_triangle_vertices)
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
    Create orthonormal coordinate system from triangle.
    triangle_vertices: (3, 3) array of vertices
    Returns: (rotation_matrix, origin)
    """
    v1, v2, v3 = triangle_vertices
    
    # Edge vectors
    edge1 = v2 - v1
    edge2 = v3 - v1
    
    # Z-axis: normal to plane
    z_axis = np.cross(edge1, edge2)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-10:
        # Degenerate triangle, use default axes
        return np.eye(3), v1
    
    z_axis = z_axis / z_norm
    
    # X-axis: along first edge
    x_axis = edge1
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-10:
        # Edge too short, use arbitrary perpendicular vector
        if abs(z_axis[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, temp)
        x_norm = np.linalg.norm(x_axis)
    
    x_axis = x_axis / x_norm
    
    # Y-axis: perpendicular to X and Z
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-10:
        # Should not happen with proper normalization
        y_axis = np.array([0.0, 0.0, 1.0])
    else:
        y_axis = y_axis / y_norm
    
    # Rotation matrix (columns are basis vectors)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    return rotation_matrix, v1


def compute_convex_hull_2d(points):
    """
    Compute 2D convex hull of points using SciPy.
    points: (n, 2) array of 2D points
    Returns: hull_points (m, 2) array of hull vertices in counterclockwise order
    """
    if len(points) < 3:
        return points
    
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except:
        # Fallback: return original points if hull fails
        return points


def rotating_calipers_min_area_rectangle(points):
    """
    Find minimum area bounding rectangle using rotating calipers algorithm.
    points: (n, 2) array of convex hull points in CCW order
    Returns: (min_area, rotation_angle, width, height)
    """
    n = len(points)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    # Initialize calipers
    min_area = float('inf')
    best_angle = 0.0
    best_width = 0.0
    best_height = 0.0
    
    # For each edge as base direction
    for i in range(n):
        # Current edge direction
        p1 = points[i]
        p2 = points[(i + 1) % n]
        edge_dir = p2 - p1
        edge_len = np.linalg.norm(edge_dir)
        
        if edge_len < 1e-10:
            continue
        
        # Unit direction vector
        u = edge_dir / edge_len
        
        # Perpendicular vector
        v = np.array([-u[1], u[0]])
        
        # Project all points onto u and v axes
        proj_u = np.dot(points, u)
        proj_v = np.dot(points, v)
        
        # Calculate extents
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
    Optimized bounding box calculation using convex hull and rotating calipers.
    """
    if vertices.shape[0] == 0:
        return np.zeros(3), np.zeros(3), np.eye(3)
    
    # Normalize normal
    z_axis = normal_vector / np.linalg.norm(normal_vector)
    
    # Extract all unique vertices (much faster than original)
    # Reshape to (n*3, 3) and use unique
    all_vertices = vertices.reshape(-1, 3)
    
    # Use rounding to reduce duplicates (faster than set of tuples)
    rounded_vertices = np.round(all_vertices, decimals=6)
    unique_vertices = np.unique(rounded_vertices, axis=0)
    
    # Project vertices onto plane perpendicular to normal
    # Vectorized projection
    v_translated = unique_vertices - origin
    projection_lengths = np.dot(v_translated, z_axis)
    
    # Use broadcasting for efficiency
    projection_lengths_3d = projection_lengths[:, np.newaxis]
    projected_2d = v_translated - projection_lengths_3d * z_axis
    
    # Compute 2D convex hull
    hull_points_2d = compute_convex_hull_2d(projected_2d[:, :2])  # Use only x,y
    
    if len(hull_points_2d) == 0:
        return np.zeros(3), np.zeros(3), np.eye(3)
    
    # Find minimal rectangle using rotating calipers
    min_area, angle, width, height = rotating_calipers_min_area_rectangle(hull_points_2d)
    
    # Create rotation matrix for optimal orientation
    # Base axes in plane
    temp = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_base = np.cross(z_axis, temp)
    x_base = x_base / np.linalg.norm(x_base)
    y_base = np.cross(z_axis, x_base)
    y_base = y_base / np.linalg.norm(y_base)
    
    # Optimal rotation in plane
    x_axis = np.cos(angle) * x_base + np.sin(angle) * y_base
    y_axis = -np.sin(angle) * x_base + np.cos(angle) * y_base
    
    # Final rotation matrix
    best_rotation = np.column_stack([x_axis, y_axis, z_axis])
    
    # Transform all vertices to find extents
    transformed = np.dot(unique_vertices - origin, best_rotation)
    min_coords = transformed.min(axis=0)
    max_coords = transformed.max(axis=0)
    
    return min_coords, max_coords, best_rotation


# ========== Original API compatibility ==========

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
    # Use optimized implementation
    vertices, triangles = parse_stl_vectorized(filename)
    total_triangles = vertices.shape[0]
    
    if total_triangles == 0:
        return None
    
    # Find max area triangle (vectorized)
    max_area, max_index, max_triangle_vertices = find_max_area_triangle_vectorized(vertices)
    
    # Create coordinate system from max triangle
    v1, v2, v3 = max_triangle_vertices
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal_vector = np.cross(edge1, edge2)
    
    # Calculate aligned bounding box (optimized)
    min_coords, max_coords, rotation_matrix = calculate_aligned_bounding_box_optimized(
        vertices, normal_vector, v1
    )
    
    dimensions = max_coords - min_coords
    volume = dimensions[0] * dimensions[1] * dimensions[2]
    
    # Create compatibility dict
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
