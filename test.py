import cv2
import pyscreenshot

im = pyscreenshot.grab(bbox=(10, 10, 1800, 1800))
im.save("tmp.png")

#image = cv2.imread("tmp.png")
#cv2.imshow("Input", image)
#cv2.waitKey(5000)