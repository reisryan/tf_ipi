from main import BinarizeImage
import time
import os
import subprocess as sp

waittime = 20   # tempo para esperar entre rodagens do avaliador.  Usado para alinhar o output de print com o do terminal

starttime = time.time()
for m in range(1,11): 
    print("Processing image: ",m)
    try:
        BinarizeImage(m,False)
        dig1 = m // 10
        dig2 = m % 10
        print("Globalizada: ",m)
        sp.Popen(f'./Avaliacao/DIBCO_metrics.exe ./Avaliacao/GT/H{dig1}{dig2}_estGT.tiff ./resultado_HDIBCO/H{dig1}{dig2}_GPos_Processada.png ./Avaliacao/GT/H{dig1}{dig2}_estGT_RWeights.dat ./Avaliacao/GT/H{dig1}{dig2}_estGT_PWeights.dat')
        time.sleep(waittime)
        print("Nick: ",m)
        sp.Popen(f'./Avaliacao/DIBCO_metrics.exe ./Avaliacao/GT/H{dig1}{dig2}_estGT.tiff ./resultado_HDIBCO/H{dig1}{dig2}_NPos_Processada.png ./Avaliacao/GT/H{dig1}{dig2}_estGT_RWeights.dat ./Avaliacao/GT/H{dig1}{dig2}_estGT_PWeights.dat')
        time.sleep(waittime)
        print("\n")
    except:
        print("ERROR")
        pass
    
    