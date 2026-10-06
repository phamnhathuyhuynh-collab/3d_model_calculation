import trimesh
import matplotlib.pyplot as plt
import numpy as np
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

# print("PC1:", pc1)
# print("PC2", pc2) 
# print("PC3", pc3) 
#
# print("Explained variance ratio:")
# print(pca.explained_variance_ratio_)

vertices_pca = pca.transform(vertices) 

projection_l = vertices[:, 2] # length_axis 

min_index_l = np.argmin(projection_l) 
max_index_l = np.argmax(projection_l) 

front_vertices = vertices[vertices[:, 2] <= min_index_l + (max_index_l - min_index_l) * 0.5]
projection_w = front_vertices[:, 0] #width_axis 

min_index_w = np.argmin(projection_w) 
max_index_w = np.argmax(projection_w) 

point_a_l = vertices[min_index_l] 
point_b_l = vertices[max_index_l] 

point_a_w = vertices[min_index_w]
point_b_w = vertices[max_index_w] 

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

length_real = 268.5
print("Length =", length_real)
print("Width =", distance_w * length_real / distance_l)
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

normal = np.array([0.0, dz, dx])
normal = normal / np.linalg.norm(normal) 

mesh.show()
section = mesh.section(
    plane_origin=A, 
    plane_normal=normal
)

if section is None: 
    print("fcccckk")
else: 
    section.show()
    print("Length Section: ", section.length * length_real / distance_l)
