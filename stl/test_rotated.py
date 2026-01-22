import sys
sys.path.insert(0, 'c:/Users/Инженер/Practica/stl')
import stl

result = stl.calculate_parallelepiped_volume('c:/Users/Инженер/Practica/stl/20mm_cube.stl')

print(f"Max triangle index: {result['max_triangle_index']}")
print(f"Max triangle area: {result['max_triangle_area']}")
print(f"Max triangle vertices: {result['max_triangle']['vertices']}")
print(f"Dimensions: {result['dimensions']}")
print(f"Min coords: {result['min_coords']}")
print(f"Max coords: {result['max_coords']}")
print(f"Volume: {result['volume']}")
print(f"Expected: 8000")
print(f"Ratio: {result['volume'] / 8000}")
