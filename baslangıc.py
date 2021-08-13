import cv2 
import numpy as np 

resim=cv2.imread("nothing_here.jpg",0)

cv2.imshow("merdivenaltimuhendisligi",resim)

#cv2.imwrite("siyahbeyaz.png",resim)
print(resim)

cv2.waitKey(0)
cv2.destroyAllWindows()

