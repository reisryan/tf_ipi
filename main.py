import cv2 as cv
import numpy as np
from skimage.filters import threshold_multiotsu
import os
import matplotlib.pyplot as plt
import statistics


# ==============================================================================
# 0. Kernels usados no hit-miss do pós processamento
# ==============================================================================


arrGap = np.array([[-1, -1, -1],
                   [-1, 1, -1],
                   [-1, -1, -1]], dtype=np.float32)  # Array de gap

arrDot = np.array([[1, 1, 1],
                   [1, -1, 1],
                   [1, 1, 1]], dtype=np.float32)  # Array de detectar

arrconvU = np.array([
    [0, 0, 0, 0, 0,],
    [0, 1, 1, 1, 0],
    [0, 1, -1, 1, 0],
    [0, -1, -1, -1, 0],
    [0, -1, -1, -1, 0]], dtype=np.float32)  # Array de detectar ponto

arrconvD = np.array([
    [0, -1, -1, -1, 0],
    [0, -1, -1, -1, 0],
    [0, 1, -1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0,]], dtype=np.float32)  # Array de detectar ponto


arrconvL = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, -1, -1],
    [0, 1, -1, -1, -1],
    [0, 1, 1, -1, -1],
    [0, 0, 0, 0, 0]], dtype=np.float32)  # Array de detectar ponto

arrconvR = np.array([
    [0, 0, 0, 0, 0],
    [-1, -1, 1, 1, 0],
    [-1, -1, -1, 1, 0],
    [-1, -1, 1, 1, 0],
    [0, 0, 0, 0, 0]], dtype=np.float32)  # Array de detectar ponto

arrconv = [arrconvU, arrconvD, arrconvL, arrconvR]

arrconcU = np.array([
    [0, 0, 0, 0, 0,],
    [0, 1, 1, 1, 0],
    [0, -1, 1, -1, 0],
    [0, -1, -1, -1, 0],
    [0, -1, -1, -1, 0]], dtype=np.float32)  # Array de detectar ponto

arrconcD = np.array([
    [0, -1, -1, -1, 0],
    [0, -1, -1, -1, 0],
    [0, -1, 1, -1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0,]], dtype=np.float32)  # Array de detectar ponto


arrconcL = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, -1, -1, -1],
    [0, 1, 1, -1, -1],
    [0, 1, -1, -1, -1],
    [0, 0, 0, 0, 0]], dtype=np.float32)  # Array de detectar ponto

arrconcR = np.array([
    [0, 0, 0, 0, 0],
    [-1, -1, 1, 1, 0],
    [-1, -1, -1, 1, 0],
    [-1, -1, 1, 1, 0],
    [0, 0, 0, 0, 0]], dtype=np.float32)  # Array de detectar ponto

arrconc = [arrconcU, arrconcD, arrconcL, arrconcR]

# ==============================================================================
# 1. CRIAÇÃO DA PASTA DE RESULTADOS E LEITURA DA IMAGEM
# ==============================================================================
# Configuração do diretório de saída para salvar os resultados intermediários e finais.
output_dir = './resultados_binarizacao_nick'
os.makedirs(output_dir, exist_ok=True)

image_path = './DIBC02009_Test_images-handwritten/H04.bmp'

# Leitura da imagem e conversão para escala de cinza.
# O casting para float32 (gf) é necessário para evitar estouro de memória (overflow/underflow)
# durante as operações aritméticas nas etapas de cálculo de contraste.
img = cv.imread(image_path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.uint8)
gf = gray.astype(np.float32)

# ==============================================================================
# 2. PRÉ-PROCESSAMENTO: ESTIMATIVA DE CONTRASTE LOCAL E APLICAÇÃO DE CLAHE
# ==============================================================================
# O objetivo deste bloco é avaliar se a imagem possui baixo contraste local e,
# se necessário, aplicar a equalização adaptativa de histograma para melhoria de detalhes.

# Definição do elemento estruturante (kernel 3x3) para operações morfológicas.
kernel = np.ones((3, 3), np.uint8)

# Operações morfológicas para mapear os valores extremos locais (vizinhança 3x3):
# - Dilatação (dilate): Encontra a intensidade máxima local (envelope superior).
# - Erosão (erode): Encontra a intensidade mínima local (envelope inferior).
local_max = cv.dilate(gf, kernel)
local_min = cv.erode(gf, kernel)

# Cálculo do Contraste Local de Michelson: C = (Max - Min) / (Max + Min)
# Utiliza-se um valor infinitesimal 'e' (epsilon) para evitar divisões por zero.
soma = local_max + local_min
diferenca = local_max - local_min
e = 1e-6

local_contrast_map = np.zeros_like(gf)
mask = soma > 0
local_contrast_map[mask] = diferenca[mask] / (soma[mask] + e)

# Média global do contraste de Michelson local para classificar o nível de contraste da imagem.
final_local_contrast = np.mean(local_contrast_map)

# Se o contraste médio local for muito baixo (< 0.02), aplica-se o CLAHE
# (Contrast Limited Adaptive Histogram Equalization) para realçar os gradientes locais
# de cinza, limitando a amplificação de ruído em regiões homogêneas.
if final_local_contrast < 0.02:
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

# ==============================================================================
# 3. SELEÇÃO DA LIMIARIZAÇÃO GLOBAL BASEADA EM CONTRASTE (DECISÃO DINÂMICA)
# ==============================================================================
# Aqui define-se qual algoritmo de binarização global é mais adequado com base
# no contraste local estimado. É utilizado o algoritmo Multi-Otsu (com 3 classes,
# gerando 2 limiares: lim[0] e lim[1]) e o Otsu padrão de 2 classes (TO).

T01, T02, T03 = 0.03, 0.04, 0.085
lim = threshold_multiotsu(gray, classes=3)

if final_local_contrast <= T01:
    # Cenário de contraste extremamente baixo: Adota o limiar Multi-Otsu mais alto
    # (lim[1]) para forçar a segmentação do texto fracamente definido.
    _, img_binarizada = cv.threshold(gray, lim[1], 255, cv.THRESH_BINARY)
elif final_local_contrast <= T02:
    # Cenário de contraste muito baixo: Analisa a distribuição de pixels entre o
    # limiar de Otsu clássico (TO) e o limiar Multi-Otsu superior (lim[1]).
    # Se a diferença de limiares for razoável (Dmin a Dmax) e o número de pixels
    # na faixa intermediária (nb) for pequeno em relação ao fundo (nb_1),
    # utiliza-se lim[1]; caso contrário, mantém-se o Otsu padrão (TO).
    Dmin, Dmax, P = 5, 25, 0.5
    _, thresh_otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print (_, thresh_otsu)
    TO = _
    nb = np.sum((gray > TO) & (gray <= lim[1]))
    nb_1 = np.sum(gray <= TO)
    d = abs(lim[1] - TO)

    if (Dmin <= d <= Dmax) and (nb <= (P*nb_1)):
        _, img_binarizada = cv.threshold(gray, lim[1], 255, cv.THRESH_BINARY)
    else:
        _, img_binarizada = cv.threshold(gray, TO, 255, cv.THRESH_BINARY)
elif final_local_contrast <= T03:
    # Cenário de contraste moderado: Aplica o algoritmo clássico de Otsu (limiar global ideal).
    _, img_binarizada = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
else:
    # Cenário de contraste alto: Adota o limiar Multi-Otsu inferior (lim[0])
    # para evitar a inclusão de ruídos de fundo como parte do objeto binarizado.
    _, img_binarizada = cv.threshold(gray, lim[0], 255, cv.THRESH_BINARY)

# ==============================================================================
# 4. LIMIARIZAÇÃO LOCAL ADAPTATIVA DE NICK
# ==============================================================================
# O algoritmo de NICK calcula um limiar específico para cada pixel com base nas
# estatísticas locais (média e variância) de uma vizinhança de tamanho ws x ws.
# A fórmula matemática é: T(x, y) = m(x, y) + k_nick * sqrt((sum_sq - m^2) / N)
# É ideal para preservar traços finos e contornos de escrita manual (tinta).

ws = 7
k_nick = -0.05  # Parâmetro de controle da influência do desvio padrão local.
gray_f = gray.astype(np.float64)

# Cálculo eficiente da média local m(x, y) usando Box Filter (filtro de caixa 2D uniforme).
m_img = cv.boxFilter(gray_f, -1, (ws, ws), normalize=True)

# Cálculo da média dos quadrados locais para computar a variância local de forma otimizada.
gray_sq = gray_f ** 2
sum_sq_img = cv.boxFilter(gray_sq, -1, (ws, ws), normalize=False)

NP = ws * ws
# Cálculo do radicando (estimador de variância/desvio padrão da vizinhança).
# Garante-se que valores levemente negativos causados por precisão numérica sejam zerados.
radicand = (sum_sq_img - (m_img ** 2)) / NP
radicand = np.maximum(radicand, 0.0)

# Geração do mapa bidimensional de limiares locais T_nick_img.
T_nick_img = m_img + k_nick * np.sqrt(radicand)

# Segmentação local: pixels com intensidade menor ou igual ao limiar local viram preto (0),
# e pixels maiores viram branco (255).
img_nick = np.zeros_like(gray, dtype=np.uint8)
img_nick[gray <= T_nick_img] = 0
img_nick[gray > T_nick_img] = 255

# ==============================================================================
# 5. DETECÇÃO DE MANCHAS (SMEAR) POR DENSIDADE E MESCLAGEM HÍBRIDA
# ==============================================================================
# Manchas ou vazamento de tinta (smear) degradam a binarização global.
# Este bloco divide a imagem binarizada global em blocos não-sobrepostos,
# calcula a densidade de pixels pretos (texto/mancha) em cada bloco e
# rotula blocos com densidade anormalmente alta (outliers) como manchas.

k_smear = 0.001
h, w = img_binarizada.shape
fSlist = []
blockscoords = []

# Partição da imagem em blocos disjuntos de dimensão ws x ws.
for y in range(0, h, ws):
    for x in range(0, w, ws):
        block = img_binarizada[y:y+ws, x:x+ws]
        tp = block.size
        if tp > 0:
            # Quantidade de pixels pretos no bloco.
            bp = np.sum(block == 0)
            fS = bp / tp                 # Fração de pixels pretos (densidade).
            fSlist.append(fS)
            blockscoords.append((y, x, fS, block.shape))

# Média e desvio padrão global da densidade de pixels pretos por bloco.
m_fS, s_fS = np.mean(fSlist), np.std(fSlist)

smear_mask = np.zeros_like(img_binarizada)

# Classificação dos blocos: se a fração de pretos exceder (média + k_smear * desvio_padrão),
# a região é classificada como mancha ou região densa de interesse.
for y, x, fS, shape in blockscoords:
    if fS > (m_fS + k_smear * s_fS):
        smear_mask[y:y+shape[0], x:x+shape[1]] = 255

# Dilatação morfológica da máscara de manchas para abranger as bordas e áreas de transição.
kernel_dilate_mask = np.ones((ws, ws), np.uint8)
smear_mask = cv.dilate(smear_mask, kernel_dilate_mask, iterations=1)

# Mesclagem híbrida (Hybrid Binarization):
# - Nas áreas cobertas pela máscara de mancha (smear_mask == 255), utiliza-se o resultado
#   do algoritmo local adaptativo (img_nick), que é mais robusto a variações locais de iluminação.
# - Nas áreas limpas de fundo (fundo normal), preserva-se o resultado da binarização global.
img_b2 = np.where(smear_mask == 255, img_nick, img_binarizada)


def HitMiss(img, shape):
    return cv.morphologyEx(img, cv.MORPH_HITMISS, shape)

# ==============================================================================
# 7. Pós processamento
# ==============================================================================


imagem_pos = cv.bitwise_not(img_b2)
imagem_inversa = imagem_pos

# Remove pixels do foreground cercados por background
dotimg = HitMiss(imagem_pos, arrDot)
imagem_pos = imagem_pos - dotimg

# Remove pixels individuais do background cercados por foreground
gapimg = HitMiss(imagem_pos, arrGap)
imagem_pos = imagem_pos + gapimg

# Encontra os componentes conexos
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
    imagem_pos, connectivity=8)


pixel_dict = {}
for label in range(num_labels):
    if (label != 0):
        pixel_dict[label] = 0

# Conta o número de pixels em cada componente
for row in labels:
    for pixel in row:
        if pixel != 0:
            pixel_dict[pixel] += 1

valuelist = []

# Valor lambda utilizado na função de detectar "manchas"
lamb = 45

# Calcula a média e desvio padrão dos componentes
values = list(pixel_dict.values())
mean = statistics.mean(values)
std_dev = statistics.stdev(values)

# Mantém apenas os componentes de áreas menores que essa expressão
keep_labels = []
for label, count in pixel_dict.items():
    if count > (lamb*mean)/std_dev:
        keep_labels.append(label)

print(pixel_dict.items(), "\n\n", mean, std_dev)
print()
print(keep_labels)

# Normaliza para 255 o que é para ser mantido,e para 0 caso contrário
for row in labels:
    for pos, pixel in enumerate(row):
        if pixel in keep_labels:
            row[pos] = 255
        else:
            row[pos] = 0

con_imagem = cv.bitwise_not(labels.astype(np.uint8))
imagem_pos = con_imagem

# Remove concavidades de 1 pixel
for conv in arrconv:
    convrem = HitMiss(imagem_pos, conv)
    imagem_pos = imagem_pos - convrem

# Remove convexidades de 1 pixel
for conc in arrconc:
    concrem = HitMiss(imagem_pos, conc)
    imagem_pos = imagem_pos + concrem


imagem_final = imagem_pos


# ==============================================================================
# 7. SALVAMENTO DOS RESULTADOS
# ==============================================================================
base_name = os.path.basename(image_path).split('.')[0]
cv.imwrite(os.path.join(output_dir, f"{base_name}_01_Original_Gray.png"), gray)
cv.imwrite(os.path.join(
    output_dir, f"{base_name}_02_Binarizada_global.png"), img_binarizada)
cv.imwrite(os.path.join(
    output_dir, f"{base_name}_03_Binarizada_Nick.png"), img_b2)
cv.imwrite(os.path.join(
    output_dir, f"{base_name}_04_Pos_Processada.png"), imagem_final)
print(f"Resultados salvos com sucesso na pasta: {output_dir}")
