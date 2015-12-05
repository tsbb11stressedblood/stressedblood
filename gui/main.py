"""
Main entry file for the GUI and the software.

author: Christoph H.
last modified: 7th November 2015
"""

from gui import *
import numpy as np

root = Tk()
root.wm_title("Stressed Blood")
# First get the resolution of the screen
x = root.winfo_screenwidth()
y = root.winfo_screenheight()

# Make the window a fixed size based on resolution, and non-resizable
w = int(x*0.8)
h = int(y*0.8)
x_pos = 50
y_pos = 50
root.geometry('%dx%d+%d+%d' % (w, h, x_pos, y_pos))
#root.geometry('{}x{}'.format(int(x*0.8), int(y*0.8)))


def on_close():
    root.destroy()
    exit(0)

# Turns off resizing
root.resizable(0, 0)
app = GUI(root)
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
"""
#imgpath1 = 'smallbloodsmear.jpg'
# imgpath2 = 'test.tif'
#img =  cv2.imread(imgpath2)
img=np.load("../gui/.npy")
rbc_seg.segmentation(img)
"""