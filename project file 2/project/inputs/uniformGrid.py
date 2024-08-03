import numpy as np

def generate_mesh(num_points, filename):
    # Generate a uniform mesh from 0 to 1
    mesh = np.linspace(0, 1, num_points)
    
    # Save the mesh to the specified file
    with open(filename, 'w') as file:
        for i, value in enumerate(mesh):
            file.write(f"{i+1:>10} {value:.7E}\n")

# Generate xgrid.dat2 and ygrid.dat2
generate_mesh(51, "xgrid.dat2")
generate_mesh(51, "ygrid.dat2")