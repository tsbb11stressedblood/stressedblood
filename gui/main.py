"""
Main entry file for the GUI and the software.

author: Christoph H.
last modified: 7th November 2015
"""

from gui import *

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
