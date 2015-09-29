
import matplotlib.pyplot as plt


I = plt.imread("bloodsmear.tif")
plt.imshow(I, interpolation='nearest')
plt.show()

print("DONE")