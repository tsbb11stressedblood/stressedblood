import os
from os.path import join, isfile

import matplotlib.image as mpimg


folder = '../learning_images/9W'

#for root, dirs, files in os.walk(folder):
    #print root, files
#    for name in files:
#        img =  mpimg.imread(os.path.join(root, name))


#files = [f for r,d,f in os.walk(folder)]

#print len(files)


import os
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames if os.path.splitext(f)[1] == '.png']

print result