"""
GUI handler for the project.

This module allows users to load images, converts them to something actually usable (not NDPI) and
displays them. Also handles ROI selection of the image that is later passed on and used in the segmentation and
classification steps.

author: Christoph H.
last modified: 4th October 2015
"""

from Tkinter import *
import tkFileDialog
import tkMessageBox
from openslide import *
import os
from PIL import ImageTk as itk
from matplotlib import pyplot as plt
import cv2


class GUI:
    def __init__(self, master):

        #First we need a frame, which serves as the basic OS-specific window
        self.frame = Frame(master)
        #Packing it means that we want to fit it snuggly to the master and to make it visible
        self.frame.pack()

        #Next, we want a button where we can load an image
        self.load_button = Button(self.frame, text="Load image", command=self.load_image)
        self.load_button.pack()

    def load_image(self):
        #Get the filepath
        filename = tkFileDialog.askopenfilename()

        #Do some error checking to see if it is good or nah
        if filename:
            #Try to load NDPI-image
            path, file_extension = os.path.splitext(filename)

            if file_extension.lower() == ".ndpi":
                print("Loading ndpi-image using openslide")
                ndpi_file = OpenSlide(filename)

                level_dims = ndpi_file.level_dimensions
                print(level_dims)

                #Try and get a PIL image and save it as PNG on same path
                rgba_im = ndpi_file.read_region((0, 0), 3, (2048, 1856))
                #rgba_im.save(path + ".png", "PNG")

                #Load with matplotlib and show it
                #img = cv2.imread(path + ".png")
                #plt.figure().add_subplot(111).imshow(img)
                #plt.title("Success! (?)")
                #plt.show()

                #Use TKs PhotoImage to show the image (needed for selecting ROIs I believe)
                im = itk.PhotoImage(rgba_im)
                label1 = Label(self.frame, image=im)
                label1.image = im
                #label1.place(x=0, y=0)
                label1.pack()
        else:
            show_error("There was a problem loading the file.")


#Very quick and dirty wrappers for displaying message popup windows
def show_msg(message):
    tkMessageBox.showinfo("Message", message)


def show_error(error):
    tkMessageBox.showerror("Error!", error)
