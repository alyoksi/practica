import argparse
import math
import os

import ezdxf
from ezdxf import bbox
from ezdxf.math import Vec3


def _sample_arc_angles(start_angle, end_angle):
    angles = [start_angle, end_angle]
    for angle in (0.0, 90.0, 180.0, 270.0):
        if start_angle <= end_angle:
            if start_angle <= angle <= end_angle:
                angles.append(angle)
        else:
            if angle >= start_angle or angle <= end_angle:
                angles.append(angle)
    return angles


def _collect_construction_points(entities):
    points = []
    for entity in entities:
        entity_type = entity.dxftype()
        if entity_type == "LINE":
            points.append(Vec3(entity.dxf.start))
            points.append(Vec3(entity.dxf.end))
        elif entity_type == "ARC":
            center = Vec3(entity.dxf.center)
            radius = entity.dxf.radius
            for angle in _sample_arc_angles(entity.dxf.start_angle, entity.dxf.end_angle):
                rad = math.radians(angle)
                points.append(Vec3(center.x + radius * math.cos(rad), center.y + radius * math.sin(rad), center.z))
        elif entity_type == "CIRCLE":
            center = Vec3(entity.dxf.center)
            radius = entity.dxf.radius
            for angle in (0.0, 90.0, 180.0, 270.0):
                rad = math.radians(angle)
                points.append(Vec3(center.x + radius * math.cos(rad), center.y + radius * math.sin(rad), center.z))
        elif entity_type == "ELLIPSE":
            try:
                tool = entity.construction_tool()
                for param in (0.0, 0.25, 0.5, 0.75):
                    points.append(Vec3(tool.point(param * math.tau)))
            except Exception:
                continue
        elif entity_type == "SPLINE":
            try:
                for pt in entity.fit_points:
                    points.append(Vec3(pt))
            except Exception:
                pass
            try:
                for pt in entity.control_points:
                    points.append(Vec3(pt))
            except Exception:
                pass
    return points


def _bbox_from_revolved_surfaces(surfaces, profile_points):
    bbox_points = []
    for surf in surfaces:
        if surf.dxftype() != "REVOLVEDSURFACE":
            continue
        axis_dir = Vec3(surf.dxf.axis_vector)
        if axis_dir.magnitude < 1e-9:
            continue
        axis_dir = axis_dir.normalize()
        axis_point = Vec3(surf.dxf.axis_point)

        ref = Vec3(1, 0, 0) if abs(axis_dir.x) < 0.9 else Vec3(0, 1, 0)
        v_axis = axis_dir.cross(ref).normalize()
        w_axis = axis_dir.cross(v_axis).normalize()

        min_u = None
        max_u = None
        max_r = 0.0
        for pt in profile_points:
            rel = Vec3(pt) - axis_point
            u = axis_dir.dot(rel)
            v = v_axis.dot(rel)
            w = w_axis.dot(rel)
            if min_u is None:
                min_u = u
                max_u = u
            else:
                min_u = min(min_u, u)
                max_u = max(max_u, u)
            max_r = max(max_r, math.hypot(v, w))

        if min_u is None:
            continue

        for u in (min_u, max_u):
            for v in (-max_r, max_r):
                for w in (-max_r, max_r):
                    bbox_points.append(axis_point + axis_dir * u + v_axis * v + w_axis * w)

    return bbox.BoundingBox(bbox_points) if bbox_points else None


def get_concat_bbox(path, layer=None):
    # Если путь относительный, считаем его относительно текущей папки скрипта
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)

    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    # Отделяем 3D поверхности от 2D/строительных объектов
    surface_entities = []
    construction_entities = []
    non_construction_entities = []
    all_entities = []

    for e in msp:
        if layer and e.dxf.layer != layer:
            continue
        all_entities.append(e)

        entity_type = e.dxftype()

        # 3D объекты
        if entity_type in [
            "3DSOLID",
            "SURFACE",
            "REVOLVEDSURFACE",
            "EXTRUDEDSURFACE",
            "LOFTEDSURFACE",
            "SWEPTSURFACE",
            "BODY",
            "REGION",
        ]:
            surface_entities.append(e)
            non_construction_entities.append(e)
        # 2D строительная геометрия (профили для 3D поверхностей)
        elif entity_type in ["LINE", "ARC", "CIRCLE", "SPLINE", "ELLIPSE"]:
            construction_entities.append(e)
        else:
            # Прочие сущности (3D полилинии, меши и т.д.)
            non_construction_entities.append(e)

    if not all_entities:
        return None

    # Сначала пытаемся взять bbox 3D поверхностей
    cache = bbox.Cache()
    ext = None

    if surface_entities:
        try:
            ext = bbox.extents(surface_entities, cache=cache)
            if ext and ext.has_data and abs(ext.size.z) > 1e-6:
                return ext
        except Exception as exc:
            print(f"Предупреждение: не удалось посчитать габариты по 3D поверхностям: {exc}")

    # Если 3D bbox не получился, пробуем все не-строительные сущности
    if non_construction_entities and non_construction_entities != surface_entities:
        try:
            ext = bbox.extents(non_construction_entities, cache=cache)
        except Exception as exc:
            print(f"Предупреждение: ошибка при обработке не-строительных сущностей: {exc}")

    # Последняя попытка: все сущности, но только если нет 3D поверхностей
    if (not ext or not ext.has_data) and not surface_entities:
        try:
            ext = bbox.extents(all_entities, cache=cache)
        except Exception as exc:
            print(f"Предупреждение: не удалось посчитать габариты по всем сущностям: {exc}")
            return None

    # Для 3D поверхностей попробуем оценку по профилю вращения
    if surface_entities and (not ext or not ext.has_data or abs(ext.size.z) < 1e-6):
        profile_points = _collect_construction_points(construction_entities)
        if profile_points:
            ext = _bbox_from_revolved_surfaces(surface_entities, profile_points)
            if ext and ext.has_data:
                return ext

    # Если всё ещё нет корректного bbox, пробуем заголовок
    if surface_entities and (not ext or not ext.has_data or abs(ext.size.z) < 1e-6):
        try:
            header_extmin = doc.header.get("$EXTMIN", None)
            header_extmax = doc.header.get("$EXTMAX", None)

            if header_extmin and header_extmax:
                ext = bbox.BoundingBox([Vec3(header_extmin), Vec3(header_extmax)])
        except Exception as exc:
            print(f"Предупреждение: не удалось использовать габариты из заголовка: {exc}")

    return ext


def get_bounding_box_dimensions(path, layer=None):
    """Вернуть размеры габаритного параллелепипеда (X, Y, Z) для DXF."""
    ext = get_concat_bbox(path, layer)
    if ext is None or not ext.has_data:
        return None

    return ext.size.x, ext.size.y, ext.size.z


def bounding_rect_area(path, layer=None):
    ext = get_concat_bbox(path, layer)
    if ext is None or not ext.has_data:
        return 0.0

    # Для 2D прямоугольника используем X и Y
    width = ext.size.x
    height = ext.size.y

    print(f"Размеры 2D: ширина={width}, высота={height}")
    return width * height


def bounding_volume(path, layer=None):
    ext = get_concat_bbox(path, layer)
    if ext is None or not ext.has_data:
        return 0.0

    # Для 3D параллелепипеда используем X, Y и Z
    width = ext.size.x
    height = ext.size.y
    depth = ext.size.z

    print(f"Размеры 3D: ширина={width}, высота={height}, глубина={depth}")

    return width * height * depth


def main():
    parser = argparse.ArgumentParser(description="Расчёт площади и объёма по DXF")
    parser.add_argument(
        "path",
        nargs="?",
        default="test3D.dxf",
        help="Путь к DXF файлу (по умолчанию test3D.dxf)",
    )
    args = parser.parse_args()

    area = bounding_rect_area(args.path)
    volume = bounding_volume(args.path)

    print(f"Площадь описывающего прямоугольника (2D): {area}")
    print(f"Объём описывающего параллелепипеда (3D): {volume}")


if __name__ == "__main__":
    main()
