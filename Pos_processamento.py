import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import statistics
from sklearn.cluster import KMeans

testimg = np.array([    [0, 0, 0, 0, 0, 0, 1, 0], 
                        [0, 1, 0, 0, 0, 0, 1, 1],  
                        [0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 1, 1, 1, 0], 
                        [1, 1, 1, 0, 1, 0, 1, 0], 
                        [1, 0, 1, 0, 1, 1, 1, 0], 
                        [1, 1, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0], 
                        [1, 1, 0, 0, 0, 1, 1, 1], 
                        [1, 1, 1, 0, 0, 0, 1, 1], 
                        [1, 1, 0, 0, 0, 1, 1, 1], 
                        ], dtype=np.float32)  # Array de detectar ponto

testimg2 = np.array([    [0, 1, 0, 0, 0, 0, 0, 0], 
                        [0, 1, 0, 1, 1, 0, 1, 1],  
                        [0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 1, 0], 
                        [1, 1, 0, 0, 1, 0, 1, 0], 
                        [0, 0, 0, 0, 1, 0, 0, 0], 
                        [1, 1, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0], 
                        [1, 1, 1, 0, 0, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 1, 1], 
                        ], dtype=np.float32)  # Array de detectar ponto

arrGap = np.array([[1, 1, 1], 
                        [1, -1, 1], 
                        [1, 1, 1]], dtype=np.float32)  # Array de gap

arrDot = np.array([[-1, -1, -1], 
                    [-1, 1, -1], 
                    [-1, -1, -1]], dtype=np.float32)  # Array de detectar 

arrconvU =  np.array([
                    [0, 0, 0, 0, 0,],
                    [0,-1,-1,-1, 0], 
                    [0,-1, 1,-1, 0], 
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0]], dtype=np.float32)  # Array de detectar ponto

arrconvD =  np.array([
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0], 
                    [0,-1, 1,-1, 0], 
                    [0,-1,-1,-1, 0],
                    [0, 0, 0, 0, 0,]], dtype=np.float32)  # Array de detectar ponto


arrconvL =  np.array([
                    [0, 0, 0, 0, 0],
                    [0,-1,-1, 1, 1], 
                    [0,-1, 1, 1, 1], 
                    [0,-1,-1, 1, 1],
                    [0, 0, 0, 0, 0]], dtype=np.float32)  # Array de detectar ponto

arrconvR =  np.array([
                    [0, 0, 0, 0, 0],
                    [1, 1,-1,-1, 0], 
                    [1, 1, 1,-1, 0], 
                    [1, 1,-1,-1, 0],
                    [0, 0, 0, 0, 0]], dtype=np.float32)  # Array de detectar ponto

arrconv = [arrconvU,arrconvD,arrconvL,arrconvR]

arrconcU =  np.array([
                    [0, 0, 0, 0, 0,],
                    [0,-1,-1,-1, 0], 
                    [0, 1,-1, 1, 0], 
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0]], dtype=np.float32)  # Array de detectar ponto

arrconcD =  np.array([
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0], 
                    [0, 1,-1, 1, 0], 
                    [0,-1,-1,-1, 0],
                    [0, 0, 0, 0, 0,]], dtype=np.float32)  # Array de detectar ponto


arrconcL =  np.array([
                    [0, 0, 0, 0, 0],
                    [0,-1, 1, 1, 1], 
                    [0,-1,-1, 1, 1], 
                    [0,-1, 1, 1, 1],
                    [0, 0, 0, 0, 0]], dtype=np.float32)  # Array de detectar ponto

arrconcR =  np.array([
                    [0, 0, 0, 0, 0],
                    [1, 1,-1,-1, 0], 
                    [1, 1, 1,-1, 0], 
                    [1, 1,-1,-1, 0],
                    [0, 0, 0, 0, 0]], dtype=np.float32)  # Array de detectar ponto

arrconc = [arrconcU,arrconcD,arrconcL,arrconcR]

def HitMiss(img,shape):
    return cv2.morphologyEx(img, cv2.MORPH_HITMISS, shape)

#print("Imagem Original\n",(testimg * 255).astype(np.uint8))
plt.imshow((testimg * 255).astype(np.uint8),cmap="gray")
plt.show()

imagem = (testimg * 255).astype(np.uint8)
dotimg = HitMiss(imagem,arrDot)
##print("Dot\n", dotimg)
imagem = imagem - dotimg
gapimg = HitMiss(imagem,arrGap)
##print("Gap\n",gapimg)
imagem = imagem + gapimg
#print("Imagem processada\n", imagem)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagem, connectivity=8)

pixel_dict = {}
for label in range(num_labels):
    if (label != 0):
        pixel_dict[label] = 0

for row in labels:
    for pixel in row:
        if pixel != 0:
            pixel_dict[pixel] += 1

#print(pixel_dict)

valuelist = []

lamb = 15

values = list(pixel_dict.values())
mean = statistics.mean(values)
std_dev = statistics.stdev(values)

keep_labels = []
for label, count in pixel_dict.items():
    if count <= (lamb*mean)/std_dev:
        keep_labels.append(label)

for row in labels:
    for pos,pixel in enumerate(row):
        if pixel in keep_labels:
            row[pos] = 255
        else:
            row[pos] = 0

con_imagem = labels.astype(np.uint8)
imagem = con_imagem

#print("Imagem com grupos selecionados\n",imagem)



for conv in arrconv:
    convrem = HitMiss(imagem,conv)
    imagem = imagem - convrem

for conc in arrconc:
    concrem = HitMiss(imagem,conc)
    imagem = imagem + concrem

#print("Imagem sem concavidades\n", imagem)

imagem_final = imagem

#print("Imagem Final\n",imagem_final)
plt.imshow(imagem_final.astype(np.uint8),cmap="gray")
#plt.imsave('CerebroRemovido.jpg',tumor_removido, cmap='gray')
plt.show()
