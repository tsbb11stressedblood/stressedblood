"""
GUI handler for the project.

This module foremost provides a GUI and allows users to load images, convert them to something actually usable
(not NDPI) and display them. Also handles ROI selection of the image that is later passed on and used in the
segmentation and classification steps.

author: Christoph H.
last modified: 1st December 2015
"""

from Tkinter import *
import tkFileDialog
import tkMessageBox
from ttk import Progressbar
from openslide import *
import os
from PIL import ImageTk as itk
from PIL import Image
from matplotlib import pyplot as plt
import numpy
import cv2
from Segmentation import rbc_seg
from Classification import classer
import math


# This class handles the displaying of the results of the classification of the cells classified from the user-
# selected ROI. Dat sentence tho
# NOTE: TopLevel is Tkinters cryptic name for a new child window... yeah don't ask
class ResultDisplayer(Toplevel):
    def __init__(self, cell_list, prediction):
        Toplevel.__init__(self)
        self.wm_title("Results")

        # Names of classes
        self.classes = ["Heterophile", "Other", "Lymphocyte", "Crap"]

        # Contains the list of cells that classifier has classified
        self.cell_list = cell_list

        # Contains the enumeration of the class, index corresponds to that of cell_list
        self.pred = map(int, prediction)

        # This splits the cells into pages, may look like [ [c1_info ... c20_info], [c21_info ... c40_info], ... ] etc
        # where c1_info is a tuple containing (cell1.img, prediction[cell1])
        self.pages = []
        self.max_rows = 4
        self.max_cols = 5
        self.max_cell_per_page = self.max_cols*self.max_rows
        self.cell_boxes = []
        self.curr_page = 0

        # We need some buttons
        self.calc_button = None
        self.next_button = None
        self.prev_button = None
        self.stat_button = None
        self.apply_view_button = None
        self.drop_down_menu = None
        self.drop_down_var = StringVar()

        # And a label for displaying how many cells there are etc.
        self.number_label = None

        # This method initializes buttons and labels
        self.init_layout()

        # Now we need to initialize the actual cell boxes (results)
        self.create_pages("Default")
        self.recreate_boxes()

        # Lastly render the first page
        self.render_page(0)

        # ... And update the number_label text
        self.update_number_label()

    def init_layout(self):
        # Init the buttons
        self.calc_button = Button(self, text="Calc H/L", command=self.calc_button_callback)
        self.calc_button.grid(row=0, column=1)

        self.next_button = Button(self, text="Next page", command=self.next_button_callback)
        self.next_button.grid(row=5, column=4)

        self.prev_button = Button(self, text="Previous page", command=self.prev_button_callback)
        self.prev_button.grid(row=5, column=0)

        self.stat_button = Button(self, text="Statistics...", command=self.stat_button_callback)
        self.stat_button.grid(row=0, column=0)

        self.apply_view_button = Button(self, text="Apply", command=self.apply_view_button_callback)
        self.apply_view_button.grid(row=0, column=3, sticky=W)

        # Init the drop-down list
        self.drop_down_menu = OptionMenu(self, self.drop_down_var,
                                         "Default",
                                         self.classes[0],
                                         self.classes[1],
                                         self.classes[2],
                                         self.classes[3])
        self.drop_down_menu.grid(row=0, column=2, sticky=E)

        # Init the label
        self.number_label = Label(self, text="FOO")
        self.number_label.grid(row=0, column=4)

    def apply_view_button_callback(self):
        token = self.drop_down_var.get()
        if token == "Default":
            pass
        else:
            token = self.classes.index(token)
        self.create_pages(token)
        self.curr_page = 0
        self.update_number_label()
        self.render_page(0)

    def stat_button_callback(self):
        # Calculate some statistics for the user
        class0_total = 0
        class1_total = 0
        class2_total = 0
        class3_total = 0
        for item in self.pred:
            if str(item) is "0":
                class0_total += 1
            elif str(item) is "1":
                class1_total += 1
            elif str(item) is "2":
                class2_total += 1
            elif str(item) is "3":
                class3_total += 1
            else:
                print "Some weird class got in: " + str(item)

        str0 = "# of " + self.classes[0] + ": " + str(class0_total)
        str1 = "# of " + self.classes[1] + ": " + str(class1_total)
        str2 = "# of " + self.classes[2] + ": " + str(class2_total)
        str3 = "# of " + self.classes[3] + ": " + str(class3_total)
        show_msg("Statistics", self, str0 + "\n" + str1 + "\n" + str2 + "\n" + str3)

    def update_number_label(self):
        # Define some parameters needed
        # Loop through pages to get the total
        total = 0
        for page in self.pages:
            for cell in page:
                total += 1
        total = str(total)
        curr_lower = str(1 + self.curr_page*self.max_cell_per_page)

        # Upper limit is special case if we're on the last page
        if self.curr_page+1 is not len(self.pages):
            curr_upper = str(self.max_cell_per_page + self.curr_page*self.max_cell_per_page)
        else:
            curr_upper = total

        self.number_label["text"] = "Showing cells " + curr_lower + " - " + curr_upper + " of " + total + " total"

    def create_pages(self, token):
        self.pages = []

        # Fill pages either with any cell, or only with a certain class, that's what token is for
        if isinstance(token, str):
            counter = 1
            sub_page = []
            for ind, cell in enumerate(self.cell_list):
                if counter % self.max_cell_per_page is 0:
                    sub_page.append((cell.big_img, self.pred[ind]))
                    self.pages.append(sub_page)
                    counter = 1
                    sub_page = []
                else:
                    sub_page.append((cell.big_img, self.pred[ind]))
                    counter += 1
            if counter-1 % self.max_cell_per_page is not 0:
                self.pages.append(sub_page)

        else:
            counter = 1
            sub_page = []
            for ind, cell in enumerate(self.cell_list):
                if self.pred[ind] == token:
                    if counter % self.max_cell_per_page is 0:
                        sub_page.append((cell.big_img, self.pred[ind]))
                        self.pages.append(sub_page)
                        counter = 1
                        sub_page = []
                    else:
                        sub_page.append((cell.big_img, self.pred[ind]))
                        counter += 1
            if counter-1 % self.max_cell_per_page is not 0:
                self.pages.append(sub_page)

    def recreate_boxes(self):
        self.cell_boxes = []
        # create max_cell_per_page number of boxes, use frames for this
        for row in range(self.max_rows):
            for col in range(self.max_cols):
                fr = Frame(self, width=150, height=150, borderwidth=5)#, relief=SUNKEN)
                fr.grid(row=row+1, column=col)
                fr.pack_propagate(0)
                self.cell_boxes.append(fr)

    def render_page(self, _page):
        page = self.pages[_page]

        # First loop through the frames and check that there is nothing there
        self.recreate_boxes()

        # Loop through all cells in this particular page and draw them in the frame boxes
        for index, (cell_img, pred) in enumerate(page):
            im = Image.fromarray(cell_img)
            photo = itk.PhotoImage(im)
            im_label = Label(self.cell_boxes[index], image=photo)
            im_label.image = photo # Tragic fix for things

            # Also create a text label that tells us which class this was classified to
            class_label = Label(self.cell_boxes[index], text=self.classes[pred])
            class_label.pack(side=TOP)

            im_label.pack(side=TOP)

    def prev_button_callback(self):
        # Do some checking to see that we can actually go to prev page
        if self.curr_page is not 0:
            self.curr_page -= 1
            self.render_page(self.curr_page)
            self.update_number_label()

    def next_button_callback(self):
        # Do some checking to see that we can actually go to another page
        total_pages = len(self.pages)
        if (self.curr_page + 1) is not total_pages:
            self.curr_page += 1
            self.render_page(self.curr_page)
            self.update_number_label()

    def calc_button_callback(self):
        # Calculate
        h = 0
        l = 0
        l_identifier = self.classes.index("Lymphocyte")
        h_identifier = self.classes.index("Heterophile")

        for item in self.pred:
            if str(item) is str(l_identifier):
                l += 1
            elif str(item) is str(h_identifier):
                h += 1
        if (l == 0) and (h == 0):
            show_msg("H/L-ratio", self, "There are no lymphocytes or heterophiles.")
        elif l is 0:
            show_msg("H/L-ratio", self, "There are no lymphocytes.\nThere are " + str(h) + " heterophiles.")
        elif h is 0:
            show_msg("H/L-ratio", self, "There are no heterophiles.\nThere are " + str(l) + " lymphocytes.")
        else:
            ratio = float(h)/float(l)
            show_msg("H/L-ratio", self, "H/L: " + str(ratio))


# Class for handling displaying of images, ROI selection and running the algorithms on these ROIs.
class InteractionWindow(Canvas):
    def __init__(self, master, _ndpi_file):
        self.master = master

        # For WBC-hunt
        self.hunter = False

        # Call the super constructor
        Canvas.__init__(self, master)

        # Zoom or roi mode (defaults to roi)
        self.mode = "roi"

        # Setup the input file
        self.ndpi_file = _ndpi_file
        self.im = None              # Actual PhotoImage object of the image
        self.rgba_im = None         # This is needed as a kept reference so that we can resize image on user action

        self.current_level = None   # Scale space level currently used for viewing (for handling zoom etc)

        self.image_handle = None    # Used to delete and recreate images
        self.setup_image()

        # Stuff for box selection
        self.init_box_pos = None            # In WINDOW coordinates
        self.curr_box_bbox = None           # Also in current WINDOW coordinates
        self.last_selection_region = None   # This is in level 0 coordinates
        self.zoom_level = 0     # How many times have we zoomed?
        self.zoom_region = []   # Stack of regions in level 0 coords that tells us where the zoom is, needed for transforming

        # Since we want multiple ROIs, save them in this list
        self.roi_list = []      # Contains Triple: (ROInumber, ROI_imagedata, (bbox, [sub_rois_list]))
        self.roi_counter = 0
        self.progress = Progressbar(self, orient="horizontal", length=100, mode="determinate", value=0)
        # ProgressBar to show user what is going on. Only one active at a time

        # Bind some event happenings to appropriate methods
        self.bind('<Configure>', self.resize_image)
        self.bind('<B1-Motion>', self.select_box)
        self.bind('<Button-1>', self.set_init_box_pos)
        self.bind('<ButtonRelease-1>', self.set_box)
        self.bind('<Button-3>', self.zoom_out)

        # DEBUG
        self.counter = 0

    # Handles initial setup of the input image
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

    # Whenever all ROIs and subROIs need to be redrawn, call this
    def redraw_ROI(self):
        # Start with drawing all main (red) rois
        for num, roi_container, bbox_container in self.roi_list:
            sub_rois = bbox_container[1]
            bbox = bbox_container[0]
            if len(sub_rois) is not 0:
                for sub_roi in sub_rois:
                    self.draw_rectangle(sub_roi, "green", 5, "subroi" + str(num))
            self.draw_rectangle(bbox, "red", 4, "roi"+str(num))

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

        print "Factor: " + str(factor)
        self.current_level = self.ndpi_file.get_best_level_for_downsample(factor)
        print "Current level: " + str(self.current_level)

        # We have to make sure that we only show the zoomed-in area
        if self.last_selection_region is not None:# and self.mode is "zoom":
            self.render_status_text((100, 100), "Zooming...", 0, 50)
            # Need to transform l0 coordinates to coordinates of current_level, but only width and height!
            width, height = self.transform_to_arb_level(self.last_selection_region[1][0],
                                                        self.last_selection_region[1][1],
                                                        self.current_level)

            self.rgba_im = self.ndpi_file.read_region(self.last_selection_region[0], # The x and y needs to be in l0...
                                                      self.current_level,
                                                      (int(width), int(height)))
        else:
            self.rgba_im = self.ndpi_file.read_region((0, 0), self.current_level,
                                                      self.ndpi_file.level_dimensions[self.current_level])

        self.rgba_im = self.rgba_im.resize((new_width, new_height))

        self.im = itk.PhotoImage(self.rgba_im)
        self.delete(self.image_handle)
        self.create_image(0, 0, image=self.im, anchor=NW)
        self.clear_status_text()

    def set_init_box_pos(self, event):
        # If we are in clear mode, make sure that we handle it correctly.
        if self.mode is "clear":

            # First transform the coordinates to level 0
            (mouse_x, mouse_y) = self.transform_to_level_zero(event.x, event.y)

            # Loop through roi_list and see if we clicked on one of them
            for num, roi, bbox_container in self.roi_list:
                bbox = bbox_container[0]
                if bbox[0][0] < mouse_x < bbox[0][0]+bbox[1][0]:
                    if bbox[0][1] < mouse_y < bbox[0][1]+bbox[1][1]:
                        # Found a ROI!
                        self.delete("roi"+str(num))
                        self.delete("boxselector")
                        self.delete("subroi"+str(num))
                        self.roi_list.remove((num, roi, bbox_container))
                        break
        else:
            self.init_box_pos = (event.x, event.y)

    # From level 0 to arbitrary level coordinates
    def transform_to_arb_level(self, x_coord, y_coord, to_level):
        # We know that x and y are in level 0.
        factor = self.ndpi_file.level_downsamples[to_level]
        return float(x_coord)/factor, float(y_coord)/factor

    # From window coordinates to level 0 coordinates
    def transform_to_level_zero(self, x_coord, y_coord):

        x_percent = float(x_coord)/self.winfo_width()
        y_percent = float(y_coord)/self.winfo_height()

        if self.zoom_level is 0:
            ld = self.ndpi_file.level_dimensions[0]
            return x_percent*ld[0], y_percent*ld[1]
        else:
            ld = self.zoom_region[-1]
            return x_percent*ld[1][0]+ld[0][0], y_percent*ld[1][1]+ld[0][1]

    # Member function for clearing all ROI
    def clear_all_roi(self):
        for num, roi_container, bbox_container in self.roi_list:
            sub_rois = bbox_container[1]
            if len(sub_rois) is not 0:
                self.delete("subroi" + str(num))
            self.delete("roi"+str(num))
        self.roi_list = []
        self.roi_counter = 0
        self.delete("boxselector")

    def select_box(self, event):
        bbox = (event.x, event.y)
        self.curr_box_bbox = (self.init_box_pos, bbox)
        self.delete("boxselector")

        # Depending on the mode, we get different colors on the box
        if self.mode is "roi":
            width = self.curr_box_bbox[1][0]
            height = self.curr_box_bbox[1][1]
            topx = self.curr_box_bbox[0][0]
            topy = self.curr_box_bbox[0][1]

            l0_width, l0_height = self.transform_to_level_zero(width, height)
            l0_topx, l0_topy = self.transform_to_level_zero(topx, topy)

            # Get absolute pixel differences
            l0_width = abs(l0_width - l0_topx)
            l0_height = abs(l0_height - l0_topy)
            if (l0_width > 1000) or (l0_height > 1000):
                self.create_rectangle(self.curr_box_bbox, outline="red", tags="boxselector")
            else:
                self.create_rectangle(self.curr_box_bbox, outline="yellow", tags="boxselector")
        elif self.mode is "zoom":
            self.create_rectangle(self.curr_box_bbox, outline="green", tags="boxselector")

    # Lets us render a status bar, letting users know how far something has gotten
    def render_status_text(self, position, text, percentage, length):
        percentage = int(percentage*100.0)
        x = position[0]
        y = position[1]

        self.progress["length"] = length
        self.progress["value"] = percentage
        self.progress.place(x=x, y=y)

        self.create_text(x, y, text=text, anchor=SW, font=("Purisa", 16), tags="progresstext", fill="white")

        self.master.update()

    def clear_status_text(self):
        self.progress.place_forget()
        self.delete("progresstext")
        self.master.update()

    def set_box(self, event):
        # This is executed when the user has dragged a selection box (either zoom or ROI) and he/she has now let go of
        # the mouse. Our goal here is to get the level 0 coordinates of this box and then pass this box on depending
        # on what we want to do.
        width = self.curr_box_bbox[1][0]
        height = self.curr_box_bbox[1][1]
        topx = self.curr_box_bbox[0][0]
        topy = self.curr_box_bbox[0][1]

        l0_width, l0_height = self.transform_to_level_zero(width, height)
        l0_topx, l0_topy = self.transform_to_level_zero(topx, topy)

        # Get absolute pixel differences
        l0_width = abs(l0_width - l0_topx)
        l0_height = abs(l0_height - l0_topy)

        # We need to be able to access this region in the resize function
        self.last_selection_region = [(int(l0_topx), int(l0_topy)), (int(l0_width), int(l0_height))]

        #entire_roi = numpy.array(self.ndpi_file.read_region((int(l0_topx), int(l0_topy)), 0, (int(l0_width), int(l0_height))), dtype=numpy.uint8)
        # Multiply percentages of totals and transform to high res level
        # ------------------NEW----------------
        # Code below is ugly, might be able to reduce it to single double loop
        # Check if the ROI is too big, in that case split it and put into list
        box = []
        sub_rois = []   # Needed for drawing the subrois later on

        if not self.hunter:
            """
            if self.mode is "roi" and (l0_width > 1000 or l0_height > 1000):
                fixed_size = 500.0
                no_of_x_subrois = math.floor(float(l0_width)/float(fixed_size))
                no_of_y_subrois = math.floor(float(l0_height)/float(fixed_size))
                total = (no_of_x_subrois+1.0)*(no_of_y_subrois+1.0)   # The total number of subrois needed to be done
                counter = 0                                         # Used for knowing how far we've gotten in the loops

                for curr_x in range(int(no_of_x_subrois)):
                    for curr_y in range(int(no_of_y_subrois)):
                        curr_topx = l0_topx + fixed_size*float(curr_x)
                        curr_topy = l0_topy + fixed_size*float(curr_y)

                        box.append(numpy.array(self.ndpi_file.read_region((int(curr_topx), int(curr_topy)), 0, (int(fixed_size), int(fixed_size))), dtype=numpy.uint8))
                        # For now, just print boxes to show where we cut it
                        roi = [(int(curr_topx), int(curr_topy)), (int(fixed_size), int(fixed_size))]
                        sub_rois.append(roi)
                        # Render a status text aswell
                        counter += 1
                        self.render_status_text((topx, topy-20), "Reading data...", float(counter)/total, width-topx)

                # Now we need to handle the rest of the ROI that didn't fit perfectly into the fixed_size boxes
                # Remember, this also sort of needs to loop
                # Idea: Fix x or y, loop through the other one

                # Start with looping over x, y is fixed
                topy_rest = float(l0_topy) + no_of_y_subrois*fixed_size
                height_rest = (float(l0_topy) + float(l0_height)) - float(topy_rest)
                for curr_x in range(int(no_of_x_subrois)):
                    curr_topx = l0_topx + fixed_size*float(curr_x)
                    box.append(numpy.array(self.ndpi_file.read_region((int(curr_topx), int(topy_rest)), 0, (int(fixed_size), int(height_rest))), dtype=numpy.uint8))

                    roi = [(int(curr_topx), int(topy_rest)), (int(fixed_size), int(height_rest))]
                    sub_rois.append(roi)
                    # Render a status text aswell
                    counter += 1
                    self.render_status_text((topx, topy-20), "Reading data...", float(counter)/total, width-topx)

                # Now loop over y and x is fixed
                topx_rest = float(l0_topx) + no_of_x_subrois*fixed_size
                width_rest = (float(l0_topx) + float(l0_width)) - float(topx_rest)
                for curr_y in range(int(no_of_y_subrois)):
                    curr_topy = l0_topy + fixed_size*float(curr_y)
                    box.append(numpy.array(self.ndpi_file.read_region((int(topx_rest), int(curr_topy)), 0, (int(width_rest), int(fixed_size))), dtype=numpy.uint8))

                    roi = [(int(topx_rest), int(curr_topy)), (int(width_rest), int(fixed_size))]
                    sub_rois.append(roi)
                    # Render a status text aswell
                    counter += 1
                    self.render_status_text((topx, topy-20), "Reading data...", float(counter)/total, width-topx)

                # This is the last one, in the lower right corner
                topx_rest = float(l0_topx) + no_of_x_subrois*fixed_size
                topy_rest = float(l0_topy) + no_of_y_subrois*fixed_size
                width_rest = (float(l0_topx) + float(l0_width)) - float(topx_rest)
                height_rest = (float(l0_topy) + float(l0_height)) - float(topy_rest)
                roi = [(int(topx_rest), int(topy_rest)), (int(width_rest), int(height_rest))]
                sub_rois.append(roi)
                # Render a status text aswell
                counter += 1
                self.render_status_text((topx, topy-20), "Reading data...", float(counter)/total, width-topx)

                tmp = self.ndpi_file.read_region((int(topx_rest), int(topy_rest)), 0, (int(width_rest), int(height_rest)))
                box.append(numpy.array(tmp, dtype=numpy.uint8))
            """
            # Now depending on the mode, do different things
            if self.mode is "roi": # This is the case that the ROI selected is small enough to be alone
                box.append(numpy.array(self.ndpi_file.read_region((int(l0_topx), int(l0_topy)), 0, (int(l0_width), int(l0_height))), dtype=numpy.uint8))

        if self.mode is "roi":
            print "No of subboxes in roi you just selected: " + str(len(box))
            if self.hunter:
                box.append(numpy.array(self.ndpi_file.read_region((int(l0_topx), int(l0_topy)), 0, (int(l0_width), int(l0_height))), dtype=numpy.uint8))
            self.set_roi(box, sub_rois)
        elif self.mode is "zoom":
            self.zoom()
        self.delete("boxselector")
        self.clear_status_text()

    def set_roi(self, box, sub_rois):
        # Add the ROI to our list
        if not self.hunter:
            self.roi_list.append((self.roi_counter, box, (self.last_selection_region, sub_rois)))
        else:
            new_region = [self.last_selection_region[0], (500, 500)]
            self.roi_list.append((self.roi_counter, box, (new_region, sub_rois)))

        # Keep drawing the ROIs
        self.redraw_ROI()
        self.roi_counter += 1

        #self.create_text(self.curr_box_bbox[0][0], self.curr_box_bbox[0][1], text=str(len(self.roi_list)),
        #                 anchor=SW, font=("Purisa", 16), tags="roi"+str(len(self.roi_list)))

    def draw_rectangle(self, level_0_coords, outline, line_width, tag):
        # We need to transform the level 0 coords to the current window
        #factor = 1.0/self.ndpi_file.level_downsamples[self.current_level]
        top_x = level_0_coords[0][0]
        top_y = level_0_coords[0][1]
        width = level_0_coords[1][0]
        height = level_0_coords[1][1]

        # Get the percentages
        if self.zoom_level is 0:
            ld = self.ndpi_file.level_dimensions[0]
        else:
            ld = self.zoom_region[-1][1]
            top_x = top_x - self.zoom_region[-1][0][0]
            top_y = top_y - self.zoom_region[-1][0][1]

        top_x_percent = float(top_x)/ld[0]
        top_y_percent = float(top_y)/ld[1]
        width_percent = float(width)/ld[0]
        height_percent = float(height)/ld[1]

        top_x_view = top_x_percent*self.winfo_width()
        top_y_view = top_y_percent*self.winfo_height()
        width_view = width_percent*self.winfo_width()
        height_view = height_percent*self.winfo_height()

        box = [(top_x_view, top_y_view), (width_view+top_x_view, height_view+top_y_view)]

        self.create_rectangle(box, outline=outline, tags=tag, width=line_width)

    def zoom(self):
        # Set the zoom_region in level 0 coords
        x, y = self.transform_to_level_zero(self.curr_box_bbox[0][0], self.curr_box_bbox[0][1])
        width, height = self.transform_to_level_zero(self.curr_box_bbox[1][0], self.curr_box_bbox[1][1])

        self.zoom_region.append([(int(x), int(y)), (int(width-x), int(height-y))])
        self.zoom_level += 1

        self.resize_image((self.winfo_width(), self.winfo_height()))

        # We also need to make sure that the ROIs are (visually) transformed to the new zoom level
        self.redraw_ROI()

    # Right click zooms to previous zoom level
    def zoom_out(self, event):
        if self.zoom_level is not 0:
            if self.zoom_level is 1:
                self.reset_zoom()
            else:
                # Pop one off of the stack and reduce the zoom level by 1
                self.zoom_region.pop()
                self.zoom_level -= 1

                # The selection_region is used when resizing, so simulate that we just selected a region that is further
                # down the zoom stack
                self.last_selection_region = self.zoom_region[-1]
                self.resize_image((self.winfo_width(), self.winfo_height()))
        self.redraw_ROI()

    def reset_zoom(self):
        self.last_selection_region = None
        self.zoom_region = []
        self.zoom_level = 0
        self.resize_image((self.winfo_width(), self.winfo_height()))

        self.redraw_ROI()

    def run_roi(self):

        if not self.hunter:
            # Loop through the roi list and do segmentation and classification for each roi
            cell_list = []
            # Initial loop to get the total amount of rois to do, for the progress bar
            total_no_of_rois = 0
            for num, rois, bbox_container in self.roi_list:
                for roi in rois:
                    test = ImageShower(self, roi, "Before segmentation, roi number " + str(num))
                    total_no_of_rois += 1
            print "Total no of rois to do: " + str(total_no_of_rois)

            # Counter for the progress bar
            counter = 0
            for num, rois, bbox_container in self.roi_list:
                #rois = roi_container[1]
                for roi in rois:
                    tmp_list, new_roi = rbc_seg.segment_and_remove_from_roi(roi)
                    cell_list = cell_list + tmp_list

                    # Show a segmented version of the ROI, "intermediate result"
                    test = ImageShower(self, new_roi, "After segmentation, roi number " + str(num))

                    # Also make a progress bar
                    counter += 1
                    self.render_status_text((100, 100), "Running segmentation...", float(counter/float(total_no_of_rois)), 100)
            self.clear_status_text()

            # Call the classification
            if len(cell_list) is 0:
                show_msg("No WBC", self, "Only RBCs were detected in this/these roi/rois.")
            else:
                prediction = classer.predict_cells(cell_list)
                print "No of classified unknowns: " + str(len(prediction))
                #numpy.save("red_shit2.npy", self.roi_list[len(self.roi_list)-1][1][0])

                test = ResultDisplayer(cell_list, prediction)

        """testimg = cv2.imread("../npyimages/only_whites_3.png")
        testimg = cv2.cvtColor(testimg, cv2.COLOR_BGR2RGB)
        cell_list = rbc_seg.segmentation(testimg)
        prediction = classer.predict_cells(cell_list)
        test = ResultDisplayer(cell_list, prediction)
        #fff = open('../only_smeared.pik', 'w+')
        #pickle.dump(cell_list, fff)"""

        if self.hunter:
            for num, rois, bbox_container in self.roi_list:
                #rois = roi_container[1]
                for roi in rois:
                    numpy.save("../npyimages/c_test" + str(self.counter) + ".npy", roi)
                    self.counter += 1


class ImageShower(Toplevel):
    def __init__(self, master, data, title):
        Toplevel.__init__(self)
        self.master = master
        self.wm_title(title)
        self.size = (500, 500)

        # Show the image
        self.tmp = Image.fromarray(data)
        self.tmp = self.tmp.resize(self.size)
        self.im = itk.PhotoImage(self.tmp)
        self.label = Label(self, image=self.im)
        self.label.im = self.im
        self.geometry('500x500')
        self.label.pack()


# Main class for handling GUI-related things
class GUI:
    def __init__(self, master):
        # Load the classifier
        classer.load_trainer_from_file()

        # First we need a frame, which serves as the basic OS-specific window
        self.frame = Frame(master)

        # Packing it means that we want to fit it snuggly to the master and to make it visible
        self.frame.pack(fill=BOTH, expand=YES)

        # Button handles
        self.load_button = None
        self.zoom_button = None
        self.roi_sel_button = None
        self.clear_roi_button = None
        self.clear_all_roi_button = None
        self.run_button = None
        self.restore_view_button = None

        # This stores the current image on screen
        self.curr_image = None

        # We need a panel with all the buttons
        self.setup_panel()

    def setup_panel(self):
        # We want a button where we can load an image
        self.load_button = Button(self.frame, text="Load image", command=self.load_image)
        self.load_button.grid(row=1, column=0)
        #self.load_button.pack()

        # Zoom and ROI-selection radiobuttons
        radio_group = Frame(self.frame)  # Own frame for the radiobuttons
        v = StringVar()
        self.zoom_button = Radiobutton(radio_group, text="Zoom", variable=v, value="zoom", indicatoron=0,
                                       command=self.set_zoom_mode)
        self.roi_sel_button = Radiobutton(radio_group, text="ROI", variable=v, value="roi", indicatoron=0,
                                          command=self.set_roi_mode)
        self.zoom_button.grid(row=0, column=0)
        self.roi_sel_button.grid(row=0, column=1)

        # Clear ROI button
        self.clear_roi_button = Radiobutton(radio_group, text="Clear ROI", variable=v, value="clear",
                                            indicatoron=0, command=self.set_clear_mode)
        self.clear_roi_button.grid(row=0, column=2)
        radio_group.grid(row=1, column=1)

        # Clear ALL Roi button
        self.clear_all_roi_button = Button(self.frame, text="Clear all ROI", command=self.clear_all_roi)
        self.clear_all_roi_button.grid(row=2, column=1)

        # Restore view button
        self.restore_view_button = Button(self.frame, text="Reset view", command=self.restore_view)
        self.restore_view_button.grid(row=3, column=1)

        # RUN button
        self.run_button = Button(self.frame, text="Run", command=self.run_roi)
        self.run_button.grid(row=4, column=1)

        # This is to make sure that everything is fit to the frame when it expands
        for x in range(1):
            Grid.columnconfigure(self.frame, x, weight=1)
        for y in range(1):
            Grid.rowconfigure(self.frame, y, weight=1)

    # Restores the zoom level
    def restore_view(self):
        if self.curr_image is not None:
            self.curr_image.reset_zoom()

    def run_roi(self):
        if self.curr_image is not None:
            self.curr_image.run_roi()

    # Tell ViewableImage that we want to clear all ROIs
    def clear_all_roi(self):
        if self.curr_image is not None:
            self.curr_image.clear_all_roi()

    # Telling the viewableImage that we're in clear mode
    def set_clear_mode(self):
        if self.curr_image is not None:
            self.curr_image.mode = "clear"

    # Callback functions for the radiobuttons "zoom" and Roi"
    def set_roi_mode(self):
        # First check if we have a image
        if self.curr_image is not None:
            self.curr_image.mode = "roi"

    # See above
    def set_zoom_mode(self):
        if self.curr_image is not None:
            self.curr_image.mode = "zoom"

    def load_image(self):
        # Get the filepath
        filename = tkFileDialog.askopenfilename()

        # Do some error checking to see if it is good or nah
        if filename:
            # Try to load NDPI-image
            path, file_extension = os.path.splitext(filename)

            if file_extension.lower() == ".ndpi":
                print("Loading ndpi-image using openslide")
                print "Filename: " + str(filename)
                ndpi_file = OpenSlide(filename)

                # Create a ViewableImage that handles the ndpi-file further.
                # Remember to delete the old image first
                if self.curr_image is not None:
                    self.curr_image.destroy()
                self.curr_image = InteractionWindow(self.frame, ndpi_file)
                self.curr_image.grid(row=0, columnspan=4, sticky=W+E+N+S)

            else:
                show_error("Only .ndpi-images can be handled!")
        else:
            show_error("There was a problem loading the file.")


# Very quick and dirty wrappers for displaying message popup windows
def show_msg(title, _parent, message):
    tkMessageBox.showinfo(title, message, parent=_parent)


def show_error(error):
    tkMessageBox.showerror("Error!", error)
