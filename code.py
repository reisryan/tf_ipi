import cv2 as cv
import numpy as np

# INPUT IMAGE
img = cv.imread('./TF/DIBC02009_Test_images-handwritten/H01.bmp')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.uint8)

# PRE PROCESSING OF INPUT IMAGE (IF CONTRAST IS LOWER THEN 0,02)
kernel = np.ones((3, 3), np.uint8)

local_max = cv.dilate(gray, kernel)
local_min = cv.erode(gray, kernel)

# Prepara o cálculo do contraste de Michelson
soma = local_max + local_min
diferenca = local_max - local_min
e = 1e-6

local_contrast_map = np.zeros_like(gray)
mask = soma > 0 
local_contrast_map[mask] = diferenca[mask] / (soma[mask] + e)
final_local_contrast = np.mean(local_contrast_map)

if final_local_contrast < 0.02:
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

# BINARIZATION OF INPUT IMAGE OR OF PRE-PROCESSED IMAGE
#T01, T02, T03 = 0.03, 0.04, 0.085

#if final_local_contrast < T01:

#elif final_local_contrast < T02:

#else:

cv.namedWindow('Original', cv.WINDOW_NORMAL)
cv.imshow('Original', img)
cv.namedWindow('New', cv.WINDOW_NORMAL)
cv.imshow('New', gray)
cv.waitKey(0)
cv.destroyAllWindows()
