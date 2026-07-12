import cv2 as cv
import numpy as np
from skimage.filters import threshold_multiotsu
import math

# INPUT IMAGE
img = cv.imread('./TF/DIBC02009_Test_images-handwritten/H01.bmp')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.uint8)
gf = gray.astype(np.float32)

# PRE PROCESSING OF INPUT IMAGE (IF CONTRAST IS LOWER THEN 0,02)
kernel = np.ones((3, 3), np.uint8)

local_max = cv.dilate(gf, kernel)
local_min = cv.erode(gf, kernel)

# Prepara o cálculo do contraste de Michelson
soma = local_max + local_min
diferenca = local_max - local_min
e = 1e-6

local_contrast_map = np.zeros_like(gf)
mask = soma > 0 
local_contrast_map[mask] = diferenca[mask] / (soma[mask] + e)
final_local_contrast = np.mean(local_contrast_map)

if final_local_contrast < 0.02:
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

# BINARIZATION OF INPUT IMAGE OR OF PRE-PROCESSED IMAGE
T01, T02, T03 = 0.03, 0.04, 0.085
lim = threshold_multiotsu(gray, classes=3)

if final_local_contrast <= T01:
    _, img_binarizada = cv.threshold(gray, lim[1], 255, cv.THRESH_BINARY)
elif final_local_contrast <= T02:
    Dmin, Dmax, P = 5, 25, 0.5
    _, thresh_otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    TO = thresh_otsu
    nb = np.sum((gray > TO) & (gray <= lim[1]))
    nb_1 = np.sum(gray <= TO)
    d = abs(lim[1] - TO)

    if (Dmin <= d <= Dmax) and (nb <= (P*nb_1)):
        _, img_binarizada = cv.threshold(gray, lim[1], 255, cv.THRESH_BINARY)
    else:
        _, img_binarizada = cv.threshold(gray, TO, 255, cv.THRESH_BINARY)
elif final_local_contrast <= T03:
    _, img_binarizada = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
else:
    _, img_binarizada = cv.threshold(gray, lim[0], 255, cv.THRESH_BINARY)

k = 0.8, ws = 35
k_nick = -0,2
h,w = img_binarizada.shape
fSlist = []
blockscoords = []

for y in range(0, h, ws):
    for x in range(0, w, ws):
        block = img_binarizada[y:y+ws, x:x+ws]
        tp = block.size
        bp = np.sum(block == 0)
        fS = bp/tp
        fSlist.append(fS)
        blockscoords.append((y, x, fS, block.shape))

m, s = np.mean(fSlist), np.std(fSlist)

img_b2 = img_binarizada.copy()
for y, x, fS, shape in blockcoords:
    if fS > (m + k*s):
        bg = gray[y:y+shape[0], x:x+shape[1]].astype(np.float32)

cv.namedWindow('Original', cv.WINDOW_NORMAL)
cv.imshow('Original', img)
cv.namedWindow('New', cv.WINDOW_NORMAL)
cv.imshow('New', gray)
cv.namedWindow('Binarizada', cv.WINDOW_NORMAL)
cv.imshow('Binarizada', img_binarizada)
cv.waitKey(0)
cv.destroyAllWindows()
