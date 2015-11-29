
import numpy as np
import matplotlib.pyplot as plt

from Tkinter import *
from PIL import Image, ImageTk

# class Application(Frame):
#
#     def __init__(self, master):
#         Frame.__init__(self, master)
#         self.master = master
#         self.create_widgets()
#         self.click_counter = 0
#
#     def create_widgets(self):
#         self.pack(fill=BOTH, expand=1)
#         self.button = Button(self, text = "HEJ", command=self.show_image)
#         self.quitButton = Button(self, text = "Quit", command=self.client_exit)
#         self.button.grid()
#         self.quitButton.grid()
#
#     def client_exit(self):
#         exit()
#
#     def show_image(self):
#
#         img = np.load("white_8.npy")
#         img = Image.fromarray(img)
#         photo = ImageTk.PhotoImage(img)
#         label = Label(image=photo)
#
#         if self.click_counter <= 0:
#             label.image = photo # keep a reference!
#             label.place(x=50,y=50)
#         else:
#             img = np.load("white_" + str(self.click_counter) + ".npy")
#             img = Image.fromarray(img)
#             photo = ImageTk.PhotoImage(img)
#             label.configure(image=photo)
#             label.image = photo # keep a reference!
#             label.place(x=50,y=50)
#
#
#         img = np.load("white_10.npy")
#         self.click_counter += 1
#
# root = Tk()
# root.title("SHOWER")
# root.geometry("500x500")
# app = Application(root)
# root.mainloop()






for i in range(1, 21):
    WBC_array = np.load("white_" + str(i) + ".npy")
    plt.figure(1)
    plt.subplot(450 + i)
    plt.imshow(WBC_array)
    #plt.title("white_" + str(i) + ".npy")
plt.show(1)



