import struct
import numpy as np

# Simple version to understand the issue
def parse_stl_ascii(filename):
    triangles = []
    with open(filename, 'r') as f:
        current_triangle = {'normal': None, 'vertices': []}
        for line in f:
            line = line.strip()
            if line.startswith('facet normal'):
                parts = line.split()
                current_triangle['normal'] = [float(parts[2]), float(parts[3]), float(parts[4])]
            elif line.startswith('vertex'):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                current_triangle['vertices'].append(vertex)
            elif line.startswith('endfacet'):
                if len(current_triangle['vertices']) == 3:
                    triangles.append(current_triangle)
                current_triangle = {'normal': None, 'vertices': []}
    return triangles

triangles = parse_stl_ascii('c:/Users/Инженер/Practica/stl/20mm_cube.stl')
print(f"Triangles: {len(triangles)}")

# Collect unique vertices
unique_verts = set()
for t in triangles:
    for v in t['vertices']:
        unique_verts.add(tuple(v))

print(f"Unique vertices: {len(unique_verts)}")

# Expected: 8 vertices for a cube
# (0,0,0), (20,0,0), (0,-20,0), (20,-20,0)
# (0,0,20), (20,0,20), (0,-20,20), (20,-20,20)

# Calculate bounds
verts_array = np.array([list(v) for v in unique_verts])
min_coords = np.min(verts_array, axis=0)
max_coords = np.max(verts_array, axis=0)
dims = max_coords - min_coords
volume = dims[0] * dims[1] * dims[2]

print(f"Min: {min_coords}")
print(f"Max: {max_coords}")
print(f"Dimensions: {dims}")
print(f"Volume: {volume}")
print(f"Expected: 8000 (20x20x20)")
