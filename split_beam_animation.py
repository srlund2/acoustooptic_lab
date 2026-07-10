import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import fsolve

def calculate_circle_imbalance(circle_center_x, circle_radius, line_x=0):
    """
    Calculate the difference in circle area on each side of a vertical line.
    Returns: (area_right - area_left) / total_area
    """
    # d = signed distance from line to circle center
    d = circle_center_x - line_x
    r = circle_radius
    
    if d >= r:
        # Circle is completely on the right
        return 1.0
    elif d <= -r:
        # Circle is completely on the left
        return -1.0
    
    # Circle intersects the line
    # Using the circular segment formula
    # Area on the far side of the line (away from center) = r^2 * arccos(d/r) - d * sqrt(r^2 - d^2)
    d_norm = d / r
    theta = np.arccos(np.clip(d_norm, -1, 1))
    area_far = r**2 * (theta - d_norm * np.sqrt(1 - d_norm**2))
    
    # Imbalance = (area_near - area_far) / total_area
    # where area_near + area_far = pi * r^2
    # so area_near = pi * r^2 - area_far
    # imbalance = (pi*r^2 - area_far - area_far) / (pi*r^2) = 1 - 2*area_far/(pi*r^2)
    imbalance = 1 - 2 * area_far / (np.pi * r**2)
    
    return round(imbalance,2)

def generate_circle_motion(num_frames=200):
    """
    Generate circle center positions following the motion pattern:
    center -> 25% over -> center -> 25% other side -> center
    """
    # Motion pattern: 4 segments
    quarter = num_frames // 4
    positions = []
    
    # Segment 1: center (0) -> 25% over (0.25)
    segment1 = np.linspace(0, 0.25, quarter)
    # Segment 2: 25% over -> center (0)
    segment2 = np.linspace(0.25, 0, quarter)
    # Segment 3: center -> 25% other side (-0.25)
    segment3 = np.linspace(0, -0.25, quarter)
    # Segment 4: -25% -> center (0)
    segment4 = np.linspace(-0.25, 0, num_frames - 3*quarter)
    
    positions = np.concatenate([segment1, segment2, segment3, segment4])
    
    return positions

# Setup figure and axis
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# Line parameters
line_x = 0
circle_radius = 0.5
line_height = 1.2 * (2 * circle_radius)  # 1.5 times the circle diameter
line_y_top = line_height / 2
line_y_bottom = -line_height / 2

# Draw the vertical line
line = ax.plot([line_x, line_x], [line_y_bottom, line_y_top], 'k-', linewidth=2)[0]

# Initialize circle (bright green)
circle = patches.Circle((circle_radius * 0.25, 0), circle_radius, 
                        color='lime', ec='darkgreen', linewidth=2)
ax.add_patch(circle)

# Title text for the imbalance counter
title_text = ax.text(0, 0.8, '', fontsize=16, ha='center', 
                     weight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Generate motion sequence
num_frames = 200
positions = generate_circle_motion(num_frames)
max_displacement = circle_radius * 0.60  # 25% of diameter

def animate(frame):
    # Calculate circle center position
    displacement = positions[frame] * max_displacement
    circle_x = line_x + displacement
    
    # Update circle position
    circle.center = (circle_x, 0)
    
    # Calculate imbalance
    imbalance = calculate_circle_imbalance(circle_x, circle_radius, line_x)
    
    # Update title with imbalance value
    title_text.set_text(f'Imbalance: {imbalance:.3f}')
    
    return circle, title_text

# Create animation
anim = FuncAnimation(fig, animate, frames=num_frames, interval=15, blit=True, repeat=True)

# Save as gif
writer = PillowWriter(fps=30)
anim.save('split_beam_animation.gif', writer=writer)

plt.tight_layout()
plt.show()
