import trimesh
import matplotlib.pyplot as plt
import numpy as np
from shapely import MultiPoint, concave_hull
from sklearn.decomposition import PCA

mesh = trimesh.load_mesh(
    "3381_final.obj",
    process=False,
    maintain_order=True
)

vertices = np.asarray(mesh.vertices)
# print("Vertices:", vertices.shape)

pca = PCA(n_components=3)
pca.fit(vertices)
pc1 = pca.components_[0]
pc2 = pca.components_[1]
pc3 = pca.components_[2]
vertices_pca = pca.transform(vertices)

# bắt đầu từ đây tính length, width và ball width, hiện tại độ chính xác khá ổn
projection_l = vertices[:, 2]  # length_axis

min_index_l = np.argmin(projection_l)
max_index_l = np.argmax(projection_l)

l_min = projection_l.min()
l_max = projection_l.max()

front_vertices = vertices[vertices[:, 2] >= l_min + (l_max - l_min) * 0.5]

projection_w = front_vertices[:, 0]  # width_axis

min_index_w = np.argmin(projection_w)
max_index_w = np.argmax(projection_w)

point_a_l = vertices[min_index_l]
point_b_l = vertices[max_index_l]

point_a_w = front_vertices[min_index_w]
point_b_w = front_vertices[max_index_w]

print("Point A Length:", point_a_l)
print("Point B Length:", point_b_l)

print("Point A Width:", point_a_w)
print("Point B Width:", point_b_w)

# print("Projection min:", projection_l[min_index_l])
# print("Projection max:", projection_l[max_index_l])
#
# print("Projection min:", projection_w[min_index_w])
# print("Projection max:", projection_w[max_index_w])

distance_l = np.linalg.norm(point_a_l - point_b_l)
distance_w = np.linalg.norm(point_a_w - point_b_w)

# width_real = 100
length_real = 268
# print("Length =", distance_l * width_real / distance_w)
print("distance_l: ", distance_l)
# print("Width =", width_real)
# print("Length =", distance_w * length_real / distance_l)
print("distance_w : ", distance_w)

A = []
B = []
for i in point_a_w:
    A.append(float(i))
for i in point_b_w:
    B.append(float(i))
print(A, B)
A = np.array(A)
B = np.array(B)
dx = B[0] - A[0]
dy = B[1] - A[1]
dz = B[2] - A[2]

normal = np.array([-dz, 0.0, dx])
normal = normal / np.linalg.norm(normal)

section = mesh.section(
    plane_origin=A,
    plane_normal=normal
)

# if section is None:
#     print("fcccckk")
# else:
#
#     scene = trimesh.Scene()
#
#     scene.add_geometry(mesh)
#     scene.add_geometry(section)
#
#     print("Length Section: ", section.length)


# -------------------------bắt đầu từ đây tính instep (độ khó đứng thứ 2)
instep_range_min = l_min + (l_max - l_min) * 0.3
instep_range_max = min(point_a_w[2], point_b_w[2])
instep_range = vertices[(vertices[:, 2] >= instep_range_min) & (
    vertices[:, 2] <= instep_range_max)]

C = []
D = []
for i in point_a_l:
    C.append(float(i))
for i in point_b_l:
    D.append(float(i))
E = [C[i] + (D[i] - C[i])*0.45 for i in range(3)]

normal = np.array([0.0, 0.0, 1])
normal = normal / np.linalg.norm(normal)

section_instep = mesh.section(
    plane_origin=E,
    plane_normal=normal
)
#
# if section_instep is None:
#     print("fcccckk")
# else:
#     scene.add_geometry(section_instep)
#     scene.show()
#     print("Length Section: ", section_instep.length)

# --------------------------- long heel grith bắt đầu từ đây

heel_vertices = vertices[vertices[:, 2] <= l_min + (l_max - l_min) * 0.3]
heel_cloud = trimesh.points.PointCloud(
    heel_vertices,
    colors=[255, 0, 0, 255]
)

heel_yz_vertices = heel_vertices[:, [1, 2]]
l_sole = heel_yz_vertices[:, 0]
l_heel = heel_yz_vertices[:, 1]
d_sole = heel_yz_vertices[np.argmin(l_sole)]
d_heel = heel_yz_vertices[np.argmin(l_heel)]


plt.figure(figsize=(10, 6))

plt.scatter(
    heel_yz_vertices[:, 0],
    heel_yz_vertices[:, 1],
    s=1
)

plt.scatter(
    d_sole[0],
    d_sole[1],
    c='r'
)
plt.axvline(x=d_sole[0], c='r')

plt.scatter(
    d_heel[0],
    d_heel[1],
    c='b'
)
plt.axhline(d_heel[1], c='b')

plt.scatter(
    d_sole[0],
    d_heel[1],
    c='black'
)

plt.plot(
    [d_sole[0], d_heel[0]], [d_sole[1], d_heel[1]], 'b-'
)
i_point = [d_sole[0], d_heel[1]]  # [z, y]

# xay dung mang numpy voi kich thuc (N, 2)
heel_yz_vertices_np = np.array(heel_yz_vertices)

geometry = MultiPoint(heel_yz_vertices_np)

hull = concave_hull(  # tạo đường boundary bao quanh vật thể
    geometry,
    ratio=0.2
)
# tạo mảng chức các coordinate
boundary_point = np.array(hull.exterior.coords)

# plot concave hull cho visualization
plt.plot(
    boundary_point[:, 0],
    boundary_point[:, 1],
    "ro-",
    linewidth=2
)

boundary_point_heel = boundary_point[
    (boundary_point[:, 0] <= d_heel[0]) &
    (boundary_point[:, 1] <= d_sole[1])
]

print(boundary_point_heel)
distances = np.sum(
    (boundary_point_heel - i_point)**2,
    axis=1
)

coor = boundary_point_heel[np.argmin(distances)]
print(coor)
print(i_point)

plt.scatter(
    coor[0],
    coor[1],
    c='green'
)

# print(d_min, point_boundary_final)
plt.xlabel("Chieu dai theo ban chan")
plt.ylabel("Chieu cao ban chan")
plt.title("Hien thi got chan theo 2d")

# đặt tỉ lệ của x và y bằng nhau để mà nó không hiển thị xấu, khác với thực tế
plt.axis("equal")
plt.grid(True)
plt.show()

#
# scene = trimesh.Scene()
# scene.add_geometry(heel_cloud)
# scene.show()
