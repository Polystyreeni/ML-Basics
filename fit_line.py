"""
Fits a line to a user defined dataset
User can enter points to the plot by left clicking
When done, a line will be fitted to best match the specified points

"""


def my_linfit(x, y):
    # Make a squared copy of x, since it's needed in the calculations
    x_sq = x.copy()
    for i in range(0, len(x)):
        x_sq[i] = x_sq[i] * x_sq[i]

    b = (sum(y) - (sum(np.multiply(y, x)) * sum(x)) / sum(x_sq)) / (len(x) - (sum(x) * sum(x)) / sum(x_sq))
    a = (sum(np.multiply(y, x)) - b * sum(x)) / sum(x_sq)

    return a, b

# Main
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backend_bases as bb

# Initialize x and y coordinates
x = []
y = []

# Set graph x and y limits to the ones given in the template
plt.xlim(-2, 5)
plt.ylim(0, 3)
plt.plot()

# Collect points clicked by the user
num_points = 0
while num_points < 2:
    print("Select at least 2 points (left click to select, right click to quit)")
    points = plt.ginput(-1, 0, True, bb.MouseButton.LEFT, None, bb.MouseButton.RIGHT)
    num_points = len(points)

# Populate list x and y
for point in points:
    x.append(point[0])
    y.append(point[1])

# Fit the line
a, b = my_linfit(x, y)
plt.plot(x, y, 'kx')
xp = np.arange(-2, 5, 0.1)
plt.plot(xp, a * xp + b, 'r-')
plt.draw()
print(f"My_fit: a = {a} and b = {b}")
plt.show()
