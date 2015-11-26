"""
Main entry file for the GUI and the software.

author: Christoph H.
last modified: 7th November 2015
"""

from gui import *
import numpy as np
"""
root = Tk()
# First get the resolution of the screen
x = root.winfo_screenwidth()
y = root.winfo_screenheight()

# Make the window a fixed size based on resolution, and non-resizable
root.geometry('{}x{}'.format(int(x*0.8), int(y*0.8)))

# Turns off resizing
root.resizable(0, 0)
app = GUI(root)
root.mainloop()
"""
#imgpath1 = 'smallbloodsmear.jpg'
# imgpath2 = 'test.tif'
#img =  cv2.imread(imgpath2)
img=np.load("../Classification/hard_test.npy")
rbc_seg.segmentation(img)