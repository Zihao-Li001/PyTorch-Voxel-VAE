import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def visualize_obj(file_path):
    """
    Visualize a 3D object from an OBJ file.
    
    Args:
        file_path (str): Path to the OBJ file.
    """
    # Load the OBJ file
    vertices = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = [int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]]
                faces.append(face)
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)
    
    # Plot the faces
    mesh = [[vertices[i] for i in face] for face in faces]
    faces_collection = Poly3DCollection(mesh, alpha=0.5, linewidths=1, edgecolors='r')
    ax.add_collection3d(faces_collection)

    # Better visualization
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio
    ax.view_init(elev=30, azim=45)
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 31)
    ax.set_zlim(0, 31)
    plt.title(os.path.basename(file_path))
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    for count in range(0, 24):
        # Assuming the OBJ files are named 'volume0.obj', 'volume1.obj', ..., 'volume99.obj'
        fileName = './../reconstructions/volume' + str(count) + '.obj'
        print("Visualizing:", fileName)
        visualize_obj(file_path=fileName)
        