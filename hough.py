import cv2
import numpy as np

img = cv2.imread('Hard-uvs16-uvs050719-002-3_frame_0001.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 5)
lineNumbers = 35
lines = cv2.HoughLines(edges,1,np.pi/180,lineNumbers)

the_list = [float('inf')]

for line in lines:
    for rho,theta in line:
        if abs(min(the_list, key=lambda x: abs(x - rho))-rho)<7 :
            the_list.append(rho)
            continue

        else:
            print("running")
            print(rho)
            print(theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            print(x1,y1,x2,y2)

            the_list.append(rho)
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            
cv2.imwrite('houghlines2.jpg',img)

# import cv2
# import numpy as np

# img = cv2.imread('Medium-NewZealand07-DSCN0778_frame_0295.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#
# minLineLength = 10
# maxLineGap = 10
# minNumberOfPixels = 100
# lines = cv2.HoughLinesP(gray,1,np.pi/180,minNumberOfPixels ,minLineLength,maxLineGap)
# print(len(lines))
#
# while len(lines)<8:
#     minLineLength -= 1
#     minNumberOfPixels -= 5
#     lines = cv2.HoughLinesP(gray, 1, np.pi / 180, minNumberOfPixels , minLineLength, maxLineGap)
#     print(len(lines))
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imshow('houghlines5',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
