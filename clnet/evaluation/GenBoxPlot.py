import matplotlib.pyplot as plt
import numpy as np


# Example dataset
data = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])

# Calculate statistical measures
median = np.median(data)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
mean = np.mean(data)
std = np.std(data)
iqr = q3 - q1

# Calculate the whiskers
whislo = np.min(data[data >= q1 - 1.5 * iqr])
whishi = np.max(data[data <= q3 + 1.5 * iqr])

box_data = {
    'med': median,
    'q1': q1,
    'q3': q3,
    'whislo': whislo,
    'whishi': whishi,
    'mean': mean,
    'fliers': [],
    'label': 'test'  # Add empty list for fliers (outliers)
}

# Create the figure and axis
fig, ax = plt.subplots()

# Create a custom box plot using the defined measures
ax.bxp([box_data, box_data], showmeans=True, vert=False)

# Set titles and labels
ax.set_title('Custom Box Plot')
ax.set_ylabel('Values')

# Show the plot
plt.show()
# This code does the following: