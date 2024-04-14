import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

p = np.load("transition_matrices_simple.npy")

# Create a figure and axis
fig, ax = plt.subplots()

# Create an initial empty plot
im = ax.imshow(p[:, :, 0], cmap='viridis', animated=True)

# Update function for animation
def update(frame):
    im.set_array(p[:, :, frame])  # Update the data for the plot
    ax.set_title(f'Year {frame}')  # Update the title with frame number
    return [im]

# Create the animation
ani = FuncAnimation(fig, update, frames=p.shape[-1], interval=50)

plt.colorbar(im)
# Display the animation
plt.show()