"""
Main entry testing file for the GUI.

author: Christoph H.
last modified: 4th October 2015
"""

from gui import *


root = Tk()
root.geometry('{}x{}'.format(600, 600))
app = GUI(root)
root.mainloop()
