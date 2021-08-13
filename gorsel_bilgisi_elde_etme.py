import cv2 
import numpy as np 


resim1=cv2.imread("nothing_here.jpg",0)
resim2=cv2.imread("korsan.png")


cv2.imshow("merdivenaltimuhendisligi",resim1)

#cv2.imwrite("siyahbeyaz.png",resim)


print(resim1)
print(resim1.size)
print(resim1.dtype)
print(resim1.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()

