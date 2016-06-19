from PIL import Image
img = Image.open("cutout_2.png")
im = img.load()

rs = 0
gs = 0
bs = 0
red_corners = []
green_corners = []
blue_corners = []

#store all upper left corners
for i in range(256):
	for j in range(256):
		if(im[i,j][0] != 0 and im[i-1,j-1][0] == 0 and im[i-1,j][0] == 0 and im[i,j-1][0] == 0): #red
			red_corners.append((i,j))
			rs = 1	#red square corner found

		elif(im[i,j][1] != 0 and im[i-1,j-1][1] == 0 and im[i-1,j][1] == 0 and im[i,j-1][1] == 0): #green
			green_corners.append((i,j))
			gs = 1

		elif(im[i,j][2] != 0 and im[i-1,j-1][2] == 0 and im[i-1,j][2] == 0 and im[i,j-1][2] == 0): #blue
			blue_corners.append((i,j))
			bs = 1

red_squares = []
green_squares = []
blue_squares = []

#loop through the squares to find the bottom right corner
for s in green_corners:
	x,y = s
	while im[x,y][1] == 255:
		x = x+1
		#print x
	x = x-1
	#now x is at top right corner
	while im[x,y][1] == 255:
		y = y+1
		#print y
	y = y-1
	#print s, (x,y)
	green_squares.append((s, (x,y)) )


#loop through the squares to find the bottom right corner
for s in blue_corners:
	x,y = s
	while im[x,y][2] == 255:
		x = x+1
		#print x
	x = x-1
	#now x is at top right corner
	while im[x,y][2] == 255:
		y = y+1
		#print y
	y = y-1
	#print s, (x,y)
	blue_squares.append((s, (x,y)) )

#print blue_squares

#upper left corners 
#print im[0,0]
#print red_squares, green_squares, blue_squares
#print green_squares

i = 0
for c in green_squares:
	#width = c[1][0] - c[0][0]
	#height = c[1][1] - c[0][1]
	
	#print c[1]
	crop = img.crop( (c[0][0], c[0][1], c[1][0], c[1][1]) )
	crop.save("test" + str(i) + ".png", "png")
	i = i+1

i = 0
for c in blue_squares:
	#width = c[1][0] - c[0][0]
	#height = c[1][1] - c[0][1]
	
	print c
	crop = img.crop( (c[0][0], c[0][1], c[1][0], c[1][1]) )
	crop.save("testb" + str(i) + ".png", "png")
	i = i+1
	
#print img.getbbox()

input("")