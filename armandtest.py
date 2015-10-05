
import matplotlib.pyplot as plt
import cv2

I = cv2.imread("bloodsmear.tif")# plt.imread("bloodsmear.tif")
plt.imshow(I, interpolation='nearest')
plt.show()

print("DONE")