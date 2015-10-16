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
from Segmentation import rbc_seg


# Class for handling all images that are viewable in the GUI.
class ViewableImage(Canvas):
    def __init__(self, master, _ndpi_file):
        self.master = master

        # Call the super constructor
        Canvas.__init__(self, master)

        # Zoom or roi mode (defaults to roi)
        self.mode = "roi"

        # Setup the input file
        self.ndpi_file = _ndpi_file
        self.im = None          # Actual PhotoImage object of the image
        self.rgba_im = None     # This is needed as a kept reference so that we can resize image on user action

        self.current_level = None   # Scale space level currently used for viewing (for handling zoom etc)

        self.image_handle = None  # Used to delete and recreate images
        self.setup_image()

        # Stuff for box selection
        self.init_box_pos = None
        self.curr_box_bbox = None
        self.last_selection_region = None
        self.zoom_level = 0     # How many times have we zoomed?

        # Bind some event happenings to appropriate methods
        self.bind('<Configure>', self.resize_image)
        self.bind('<B1-Motion>', self.select_box)
        self.bind('<Button-1>', self.set_init_box_pos)
        self.bind('<ButtonRelease-1>', self.set_box)
        self.bind('<Button-3>', self.reset_zoom)

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
        try:
            new_width = event.width
            new_height = event.height
        except AttributeError:
            #print "Attribute error in resize. Trying tuple variant. You are most probably in zoom mode so don't worry."
            new_width = event[0]
            new_height = event[1]

        self.master.update()  # Redraw screen so that the sizes are correct
        if self.last_selection_region is not None:
            factor = float(self.last_selection_region[1][1])/float(self.winfo_width())
        else:
            factor = float(self.ndpi_file.level_dimensions[0][0])/float(self.winfo_width())

        self.current_level = self.ndpi_file.get_best_level_for_downsample(factor)
        #print self.current_level

        # We have to make sure that we only show the zoomed-in area
        if self.last_selection_region is not None and self.mode is "zoom":
            self.rgba_im = self.ndpi_file.read_region(self.last_selection_region[0], 0,
                                                      self.last_selection_region[1])
        else:
            self.rgba_im = self.ndpi_file.read_region((0, 0), self.current_level,
                                                      self.ndpi_file.level_dimensions[self.current_level])

        self.rgba_im = self.rgba_im.resize((new_width, new_height))

        self.im = itk.PhotoImage(self.rgba_im)
        self.delete(self.image_handle)
        self.create_image(0, 0, image=self.im, anchor=NW)

    def set_init_box_pos(self, event):
        self.init_box_pos = (event.x, event.y)

    def select_box(self, event):
        bbox = (event.x, event.y)
        self.curr_box_bbox = (self.init_box_pos, bbox)
        self.delete("boxselector")

        # Depending on the mode, we get different colors on the box
        if self.mode is "roi":
            self.create_rectangle(self.curr_box_bbox, outline="yellow", tags="boxselector")
        elif self.mode is "zoom":
            self.create_rectangle(self.curr_box_bbox, outline="green", tags="boxselector")

    def set_box(self, event):
        # Get the region at best possible resolution
        # Grab these for ease of use
        width = self.curr_box_bbox[1][0]
        height = self.curr_box_bbox[1][1]
        topx = self.curr_box_bbox[0][0]
        topy = self.curr_box_bbox[0][1]

        # Get absolute pixel differences
        width = abs(width - topx)
        height = abs(height - topy)

        # This is the downsampling factor for the current level where the box was selected from
        ds = self.ndpi_file.level_downsamples[self.current_level]

        if self.zoom_level is 0:
            # This is a tuple giving the pixel size of the entire image in level where the box was selected
            ld = self.ndpi_file.level_dimensions[self.current_level]
            x_offset = 0
            y_offset = 0
        else:
            ld = self.last_selection_region[1]
            x_offset = self.last_selection_region[0][0]
            y_offset = self.last_selection_region[0][1]
            ds = 1.0

        # Now convert into percent of total
        top_x_percent = float(topx)/self.winfo_width()
        top_y_percent = float(topy)/self.winfo_height()
        width_percent = float(width)/self.winfo_width()
        height_percent = float(height)/self.winfo_height()

        # We need to be able to access this region in the resize function
        self.last_selection_region = [(int((top_x_percent*ld[0] + x_offset)*ds),
                                       int((top_y_percent*ld[1] + y_offset)*ds)),
                                      (int(width_percent*ld[0]*ds),
                                       int(height_percent*ld[1]*ds))]

        # Multiply percentages of totals and transform to high res level
        box = self.ndpi_file.read_region((int((top_x_percent*ld[0] + x_offset)*ds),
                                          int((top_y_percent*ld[1] + y_offset)*ds)),
                                         0,
                                         (int(width_percent*ld[0]*ds),
                                          int(height_percent*ld[1]*ds)))
        self.current_level = 0
        # Now depending on the mode, do different things
        if self.mode is "roi":
            self.set_roi(box)
        elif self.mode is "zoom":
            self.zoom()

    def set_roi(self, box):
        roi = numpy.array(box)
        # Call the segmentation (testing)
        rbc_seg.segmentation(roi)

    def zoom(self):
        self.zoom_level += 1
        self.resize_image((self.winfo_width(), self.winfo_height()))

    def reset_zoom(self, event):
        self.last_selection_region = None
        self.zoom_level = 0
        self.resize_image((self.winfo_width(), self.winfo_height()))


# Main class for handling GUI-related things
class GUI:
    def __init__(self, master):

        # First we need a frame, which serves as the basic OS-specific window
        self.frame = Frame(master)

        # Packing it means that we want to fit it snuggly to the master and to make it visible
        self.frame.pack(fill=BOTH, expand=YES)

        # Button handles
        self.load_button = None
        self.zoom_button = None
        self.roi_sel_button = None

        # This stores the current image on screen
        self.curr_image = None

        # We need a panel with all the buttons
        self.setup_panel()

    def setup_panel(self):
        # We want a button where we can load an image
        self.load_button = Button(self.frame, text="Load image", command=self.load_image)
        self.load_button.grid(row=1, column=0, pady=5, padx=5)
        #self.load_button.pack()

        # Zoom and ROI-selection radiobuttons
        radio_group = Frame(self.frame) # Own frame for the radiobuttons
        v = StringVar()
        self.zoom_button = Radiobutton(radio_group, text="Zoom", variable=v, value="1", indicatoron=0)
        self.roi_sel_button = Radiobutton(radio_group, text="ROI", variable=v, value="2", indicatoron=0)
        self.zoom_button.grid(row=0, column=1)
        self.roi_sel_button.grid(row=1, column=1)
        radio_group.grid(row=1, column=1)
        #self.zoom_button.pack()
        #self.roi_sel_button.pack()

        # This is to make sure that everything is fit to the frame when it expands
        for x in range(4):
            Grid.columnconfigure(self.frame, x, weight=1)
        for y in range(2):
            Grid.rowconfigure(self.frame, y, weight=1)

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
                self.curr_image.grid(row=0, columnspan=4, sticky=W+E+N+S)

            else:
                show_error("Only .ndpi-images can be handled!")
        else:
            show_error("There was a problem loading the file.")


# Very quick and dirty wrappers for displaying message popup windows
def show_msg(message):
    tkMessageBox.showinfo("Message", message)


def show_error(error):
    tkMessageBox.showerror("Error!", error)
