import cv2 as cv
import numpy as np
from skimage.filters import threshold_multiotsu
import math
import os

#Arquivo da primeira versao de binarizacao preservado

# 1. CRIAÇÃO DA PASTA DE RESULTADOS
output_dir = './resultados_binarizacao'
os.makedirs(output_dir, exist_ok=True)

image_path = './DIBC02009_Test_images-handwritten/H03.bmp'

# INPUT IMAGE
img = cv.imread(image_path)
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

k,ws = 0.8, 35
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
for y, x, fS, shape in blockscoords:
    if fS > (m + k*s):
        bg = gray[y:y+shape[0], x:x+shape[1]].astype(np.float32)

# Correção: Usar ponto em vez de vírgula para números decimais
k_nick = -0.2
h, w = img_binarizada.shape
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
for y, x, fS, shape in blockscoords:
    if fS > (m + k*s):
        # bg já está em float32 para suportar os cálculos sem estourar o limite numérico
        bg = gray[y:y+shape[0], x:x+shape[1]].astype(np.float32)
        
        # 1. Número de pixels no bloco (NP)
        NP = bg.size
        
        if NP > 0:
            # 2. Média local do bloco (m)
            m_local = np.mean(bg)
            
            # 3. Somatório dos pixels ao quadrado
            # float64 é altamente recomendado aqui para evitar perda de precisão
            sum_sq = np.sum(bg**2, dtype=np.float64)
            
            # 4. Cálculo do valor da raiz (Fórmula de Nick)
            # O max(..., 0.0) protege o código contra eventuais números negativos causados por imprecisão do float
            radicand = max((sum_sq - (m_local**2)) / NP, 0.0)
            T_nick = m_local + k_nick * np.sqrt(radicand)
            
            # 5. Aplica a binarização local no bloco usando o T_nick
            _, bin_block = cv.threshold(bg.astype(np.uint8), T_nick, 255, cv.THRESH_BINARY)
            
            # 6. Atualiza a imagem binarizada localmente
            img_b2[y:y+shape[0], x:x+shape[1]] = bin_block

# 2. SALVANDO OS RESULTADOS NA PASTA
base_name = os.path.basename(image_path).split('.')[0]
cv.imwrite(os.path.join(output_dir, f"{base_name}_01_Original_Gray.png"), gray)
cv.imwrite(os.path.join(output_dir, f"{base_name}_02_Binarizada_Global.png"), img_binarizada)
cv.imwrite(os.path.join(output_dir, f"{base_name}_03_Binarizada_Nick_Hibrida.png"), img_b2)
print(f"Resultados salvos com sucesso na pasta: {output_dir}")

cv.waitKey(0)
cv.destroyAllWindows()