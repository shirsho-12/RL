import matplotlib.pyplot as plt
import math
import torch


NUM_POINTS = 10000
points = torch.rand((NUM_POINTS, 2)) * 2 - 1            # Points made in the range (0, 2)
squared_dist = torch.pow(points, 2)
dist = squared_dist[:, 0] + squared_dist[:, 1]  # Euclidean distance squared (if num^2 <= 1, num <= 1)
dist = dist.numpy()
num_points_circle = (dist <= 1).sum()
print(num_points_circle)
points_circle = points[dist <= 1]
print(points_circle.shape)

plt.axes().set_aspect('equal')

plt.scatter(points[:, 0].numpy(), points[:, 1].numpy(), c='c')
plt.scatter(points_circle[:, 0].numpy(), points_circle[:, 1].numpy(), c='y')
i = torch.linspace(0, 2 * math.pi)
plt.plot(torch.cos(i).numpy(), torch.sin(i).numpy(), c='b')
plt.show()

pi_estimate = 4 * num_points_circle / NUM_POINTS
print("Estimated value of pi: ", pi_estimate)