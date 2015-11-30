"""
Separate GUI that lets us evaluate the performance of the segmentation

author: Christoph H.
last modified: 30th November 2015
"""

from Tkinter import *
import tkFileDialog
import tkMessageBox
from ttk import Progressbar
import os
from matplotlib import pyplot as plt
import numpy
from Segmentation import rbc_seg
from PIL import Image, ImageTk
import math


class PerfomanceMeasureGUI:
    def __init__(self, master):
        # First we need a frame, which serves as the basic OS-specific window
        self.frame = Frame(master)

        # Packing it means that we want to fit it snuggly to the master and to make it visible
        self.frame.pack(fill=BOTH, expand=YES)

        # Create a canvas where we can see the images
        self.canvas = Canvas(self.frame)
        self.canvas.grid(row=0, column=0, columnspan=4, rowspan=4, sticky=W+E+N+S)

        # Text that is static
        self.how_many_text = Label(self.frame, text="How many RBC: ")
        self.how_many_text.grid(row=5, column=0)

        # An entry for the no of cells the user sees
        self.cell_entry = Entry(self.frame)
        self.cell_entry.grid(row=5, column=1)

        # Button for confirming the choice and moving on
        self.ok_button = Button(self.frame, text="OK", command=self.ok_button_callback)
        self.ok_button.grid(row=5, column=2)

        # Init image to first one
        self.im_number = 1
        self.curr_ROI = numpy.load("../npyimages/testim_1.npy")
        self.curr_rbc_counter = None
        self.im = None

        # Init first one
        self.draw_image()

        # Store the data
        self.percentage_average = 0

    def ok_button_callback(self):
        # Processing
        # First, get the entry of user
        user_number = int(self.cell_entry.get())
        if user_number:
            curr_percent = float(self.curr_rbc_counter)/float(self.curr_rbc_counter + user_number)
            # Pretty special case
            if self.im_number is 1:
                self.percentage_average = curr_percent
            else:
                self.percentage_average = (float(curr_percent)+float(self.percentage_average))/2.0

            print "Percentage_average: " + str(self.percentage_average)

        # Update the image
        self.im_number += 1
        self.curr_ROI = numpy.load("../npyimages/testim_" + str(self.im_number) + ".npy")
        self.cell_entry.delete(0, 'end')
        self.draw_image()

    def draw_image(self):
        # Run the debug variant of the segmentation first, to get the masked ROI
        rbc_counter, masked_ROI = rbc_seg.debug_segmentation(self.curr_ROI)
        print "RBC_counter: " + str(rbc_counter)
        self.curr_rbc_counter = rbc_counter
        self.im = ImageTk.PhotoImage(Image.fromarray(masked_ROI))
        self.canvas.create_image(0, 0, image=self.im, anchor=NW)

        # Also print the current average in the top of the frame


root = Tk()
# First get the resolution of the screen
#x = root.winfo_screenwidth()
#y = root.winfo_screenheight()

# Make the window a fixed size based on resolution, and non-resizable
#root.geometry('{}x{}'.format(int(x*0.8), int(y*0.8)))

# Turns off resizing

root.resizable(0, 0)
app = PerfomanceMeasureGUI(root)
root.mainloop()


