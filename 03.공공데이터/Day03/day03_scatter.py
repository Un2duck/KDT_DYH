import matplotlib.pyplot as plt
x_value = [1, 2, 3, 4]
y_value = [10, 50, 20, 10]
size = []
for y in y_value:
    size.append(y * 5)

plt.scatter(x_value, y_value, s=size, c=range(4), cmap='jet')
plt.colorbar()
plt.show()