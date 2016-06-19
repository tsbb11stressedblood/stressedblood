import cv2
import numpy as np
im = cv2.imread('cutout.png')
im2 = im.copy()
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
img2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print contours

cv2.drawContours(im2, contours, -1, (255,255,255), 1)

i = 0
for c in contours:
	x,y,w,h = cv2.boundingRect(c)
	cv2.rectangle(im2, (x,y), (x+w, y+h), (255,0,255), 1)
	if(im2[y+1,x+1][0] == 255):
		name = 'blue'
	elif(im2[y+1,x+1][1] == 255):
		name = 'green'
	else:
		name = 'red'
		
		
	area = h*w
	img_area = np.sum(im[y:y+h, x:x+w])/(255.0)
	if area == img_area:
		cv2.imwrite(str(i) + name + '.png', im[y:y+h, x:x+w])
		i += 1
	elif im[y,x].any():	#the bounding box contains two rectangles!
		#with the first being in the upper left corner!
		#start with the first one
		newx = x
		newy = y
		while im[y,newx].any():	#if there's a pixel value at 0,0 locally
			#then find the first rectangles boundaries
			newx += 1
			#now newx is at the upper right corner
		while im[newy,x].any():	#if there's a pixel value at 0,0 locally
			#then find the first rectangles boundaries
			newy += 1
		neww, newh = newx-x, newy-y
		#save the first rectangle!
		cv2.imwrite(str(i) + name + '_.png', im[y:y+newh, x:x+neww])
		i += 1
		
		#second one:
		newx = x+w
		newy = y+h
		while im[y+h-1,newx-1].any():
			newx -= 1
		
		while im[newy-1, x+w-1].any():
			newy -= 1
		
		neww, newh = x+w-newx, y+h-newy
		print neww, newh
		#save second rectangle!
		cv2.imwrite(str(i) + name + '_x.png', im[newy:newy+newh, newx:newx+neww])
		#print newx, newy
	elif im[y,x+w-1].any():	#the bounding box contains two rectangles!
		#one at upper right corner!
		print "yeeeeesssss"
		newx = x+w
		newy = y
		while im[y,newx-1].any():
			newx -= 1
		
		while im[newy, x+w-1].any():
			newy += 1
		
		neww, newh = x+w-newx, newy-y
		print neww, newh
		#save first rectangle!
		cv2.imwrite(str(i) + name + '_y.png', im[newy-newh:newy, newx:newx+neww])
		
		#second one:
		newx = x
		newy = y+h
		while im[y+h-1,newx].any():
			newx += 1
		
		while im[newy-1, x].any():
			newy -= 1
		
		neww, newh = newx-x, y+h-newy
		print neww, newh
		#save second rectangle!
		cv2.imwrite(str(i) + name + '_z.png', im[newy:newy+newh, x:x+neww])
		

#if im2[41,52].all() == 0:
	#print "yes"

cv2.imshow('hej', im)
cv2.waitKey(0)