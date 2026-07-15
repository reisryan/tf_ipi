from main import BinarizeImage
import time
import os
import subprocess as sp

starttime = time.time()
output_dir = './resultado_HDIBCO'
os.makedirs(output_dir, exist_ok=True)

'''BinarizeImage(1)
dig1 = 1 // 10
dig2 = 1 % 10
print("\nGlobaliada: ")
sp.Popen(f'./Avaliacao/DIBCO_metrics.exe ./Avaliacao/GT/H{dig1}{dig2}_estGT.tiff ./resultado_HDIBCO/H{dig1}{dig2}_GPos_Processada.png ./Avaliacao/GT/H{dig1}{dig2}_estGT_RWeights.dat ./Avaliacao/GT/H{dig1}{dig2}_estGT_PWeights.dat')
print("\nNick: ")
sp.Popen(f'./Avaliacao/DIBCO_metrics.exe ./Avaliacao/GT/H{dig1}{dig2}_estGT.tiff ./resultado_HDIBCO/H{dig1}{dig2}_NPos_Processada.png ./Avaliacao/GT/H{dig1}{dig2}_estGT_RWeights.dat ./Avaliacao/GT/H{dig1}{dig2}_estGT_PWeights.dat')

while True:
    pass
'''

for m in range(1,11): 
    print("Processing image: ",m)
    try:
        BinarizeImage(m)
        dig1 = m // 10
        dig2 = m % 10
        print("Globalizada: ",m)
        sp.Popen(f'./Avaliacao/DIBCO_metrics.exe ./Avaliacao/GT/H{dig1}{dig2}_estGT.tiff ./resultado_HDIBCO/H{dig1}{dig2}_GPos_Processada.png ./Avaliacao/GT/H{dig1}{dig2}_estGT_RWeights.dat ./Avaliacao/GT/H{dig1}{dig2}_estGT_PWeights.dat')
        time.sleep(30)
        print("Nick: ",m)
        sp.Popen(f'./Avaliacao/DIBCO_metrics.exe ./Avaliacao/GT/H{dig1}{dig2}_estGT.tiff ./resultado_HDIBCO/H{dig1}{dig2}_NPos_Processada.png ./Avaliacao/GT/H{dig1}{dig2}_estGT_RWeights.dat ./Avaliacao/GT/H{dig1}{dig2}_estGT_PWeights.dat')
        time.sleep(30)
        print("\n")
    except:
        print("ERROR")
        pass
    
    