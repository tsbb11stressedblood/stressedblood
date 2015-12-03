from matplotlib import pyplot as plt
import numpy as np

stitched_im = np.zeros((0, 0, 3), 'uint8')
for num in range(21):
    im = np.load("../npyimages/testim_" + str(num+1) + ".npy")
    curr_size = stitched_im.shape
    new_im_size = im.shape

    # Only expand rows if we need to
    if new_im_size[0] < curr_size[0]:
        new_stitched_im = np.zeros((curr_size[0], curr_size[1]+new_im_size[1], 3), 'uint8')
        new_stitched_im[0:curr_size[0], 0:curr_size[1], :] = stitched_im
        new_stitched_im[0:new_im_size[0], curr_size[1]:new_im_size[1]+curr_size[1], :] = im[:, :, 0:3]
        stitched_im = new_stitched_im
    else:
        new_stitched_im = np.zeros((new_im_size[0], curr_size[1]+new_im_size[1], 3), 'uint8')

        new_stitched_im[0:curr_size[0], 0:curr_size[1], :] = stitched_im
        new_stitched_im[0:new_im_size[0], curr_size[1]:new_im_size[1]+curr_size[1], :] = im[:, :, 0:3]
        stitched_im = new_stitched_im

np.save("../npyimages/stitched_im.npy", stitched_im)
plt.figure()
plt.imshow(stitched_im)
plt.show()
