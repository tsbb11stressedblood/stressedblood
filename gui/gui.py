"""
GUI handler for the project.

This module allows users to load images, converts them to something actually usable (not NDPI) and
displays them. Also handles ROI selection of the image that is later passed on and used in the segmentation and
classification steps.

author: Christoph H.
last modified: 7th October 2015
"""

from Tkinter import *
import tkFileDialog
import tkMessageBox
from openslide import *
import os
from PIL import ImageTk as itk
from matplotlib import pyplot as plt
import numpy
import cv2
from Segmentation import rbcseg


# Class for handling all images that are viewable in the GUI.
class ViewableImage(Canvas):
    def __init__(self, master, _ndpi_file):
        self.master = master

        # Call the super constructor
        Canvas.__init__(self, master)

        # Pack ourselves snuggly in master
        self.pack(fill=BOTH, expand=True)

        # Setup the input file
        self.ndpi_file = _ndpi_file
        self.im = None          # Actual PhotoImage object of the image
        self.rgba_im = None     # This is needed as a kept reference so that we can resize image on user action

        self.current_level = None   # Scale space level currently used for viewing (for handling zoom etc)

        self.image_handle = None  # Used to delete and recreate images
        self.setup_image()

        # Stuff for ROI selection
        self.init_roi_pos = None
        self.curr_roi_bbox = None

        # Bind some event happenings to appropriate methods
        self.bind('<Configure>', self.resize_image)
        self.bind('<B1-Motion>', self.select_roi)
        self.bind('<Button-1>', self.set_init_roi_pos)
        self.bind('<ButtonRelease-1>', self.set_roi)

    # Hanldes initial setup of the input image
    def setup_image(self):

        # Select best scale space level and get its size

        # First get current canvas size, to determine the downsampling factor'
        self.master.update()  # Redraw screen so that the sizes are correct
        factor = float(self.ndpi_file.level_dimensions[0][0])/float(self.winfo_width())

        self.current_level = self.ndpi_file.get_best_level_for_downsample(factor)

        self.rgba_im = self.ndpi_file.read_region((0, 0), self.current_level,
                                                  self.ndpi_file.level_dimensions[self.current_level])

        self.rgba_im = self.rgba_im.resize((self.winfo_width(), self.winfo_height()))

        # Use TKs PhotoImage to show the image (needed for selecting ROIs)
        self.im = itk.PhotoImage(self.rgba_im)
        self.image_handle = self.create_image(0, 0, image=self.im, anchor=NW)

    # Method that ensures automatic resize in case the windows is resized by the user
    def resize_image(self, event):
        new_width = event.width
        new_height = event.height

        self.master.update()  # Redraw screen so that the sizes are correct
        factor = float(self.ndpi_file.level_dimensions[0][0])/float(self.winfo_width())

        self.current_level = self.ndpi_file.get_best_level_for_downsample(factor)

        self.rgba_im = self.ndpi_file.read_region((0, 0), self.current_level,
                                                  self.ndpi_file.level_dimensions[self.current_level])

        self.rgba_im = self.rgba_im.resize((new_width, new_height))

        self.im = itk.PhotoImage(self.rgba_im)
        self.delete(self.image_handle)
        self.create_image(0, 0, image=self.im, anchor=NW)

    def set_init_roi_pos(self, event):
        self.init_roi_pos = (event.x, event.y)

    def select_roi(self, event):
        bbox = (event.x, event.y)
        self.curr_roi_bbox = (self.init_roi_pos, bbox)
        self.delete("ROIselector")
        self.create_rectangle(self.curr_roi_bbox, outline="yellow", tags="ROIselector")

    def set_roi(self, event):
        # Get the ROI region at best possible resolution
        # Grab these for ease of use
        width = self.curr_roi_bbox[1][0]
        height = self.curr_roi_bbox[1][1]
        topx = self.curr_roi_bbox[0][0]
        topy = self.curr_roi_bbox[0][1]

        # Get absolute pixel differences
        width = width - topx
        height = height - topy

        # This is the downsampling factor for the current level where ROI was selected from
        ds = self.ndpi_file.level_downsamples[self.current_level]
        # This is a tuple giving the pixel size of the entire image in level where ROI was selected
        ld = self.ndpi_file.level_dimensions[self.current_level]

        # Now convert into percent of total
        top_x_percent = float(topx)/self.winfo_width()
        top_y_percent = float(topy)/self.winfo_height()
        width_percent = float(width)/self.winfo_width()
        height_percent = float(height)/self.winfo_height()

        # Multiply percentages of totals and transform to high res level
        roi = self.ndpi_file.read_region((int(top_x_percent*ld[0]*ds),
                                          int(top_y_percent*ld[1]*ds)),
                                         0,
                                         (int(width_percent*ld[0]*ds),
                                          int(height_percent*ld[1]*ds)))
        roi = numpy.array(roi)

        # Call the segmentation (testing)
        rbcseg.segmentation(roi)


# Main class for handling GUI-related things
class GUI:
    def __init__(self, master):

        # First we need a frame, which serves as the basic OS-specific window
        self.frame = Frame(master)

        # Packing it means that we want to fit it snuggly to the master and to make it visible
        self.frame.pack(fill=BOTH, expand=YES)

        # Next, we want a button where we can load an image
        self.load_button = Button(self.frame, text="Load image", command=self.load_image)
        self.load_button.pack()

        # This stores the current image on screen
        self.curr_image = None

    def load_image(self):
        # Get the filepath
        filename = tkFileDialog.askopenfilename()

        # Do some error checking to see if it is good or nah
        if filename:
            # Try to load NDPI-image
            path, file_extension = os.path.splitext(filename)

            if file_extension.lower() == ".ndpi":
                print("Loading ndpi-image using openslide")
                ndpi_file = OpenSlide(filename)

                # Create a ViewableImage that handles the ndpi-file further.
                # Remember to delete the old image first
                if self.curr_image is not None:
                    self.curr_image.destroy()
                self.curr_image = ViewableImage(self.frame, ndpi_file)
            else:
                show_error("Only .ndpi-images can be handled!")
        else:
            show_error("There was a problem loading the file.")


# Very quick and dirty wrappers for displaying message popup windows
def show_msg(message):
    tkMessageBox.showinfo("Message", message)


def show_error(error):
    tkMessageBox.showerror("Error!", error)
