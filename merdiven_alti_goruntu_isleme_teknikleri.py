
import cv2
import numpy as np


resim=cv2.imread("foto.jpg",)
cv2.imshow("farkederse kahrolayim",resim)


print("resmin matrissel içeriği :",resim)
print("resmin kaç pikselden oluşuyor: ",resim.size)

print("resmin hangi türde veri tipi ile saklandığı :",resim.dtype)
print("resmin boyutu",resim.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()

