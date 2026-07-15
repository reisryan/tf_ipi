# tf_ipi


Esse repositório armazena o código e arquivos relacionados ao projeto final do grupo composto por:
- Mateus Elias de Macedo - 222011561
- Ryan Reis Fontenele - 211036132
- Vitor Guedes Frade - 221017130

O arquivo principal é main.py, que contém uma função que binariza uma imagem de um documento para produzir um resultado mais legível. Esse arquivo pode ser executado diretamente, tendo uma função __main__ embaixo que chama a função apropriada. Para utilizar esse arquivo, você pode chamar a função BinarizeImage(código da imagem,retornar métricas). 

  Se o código < 0, processa a imagem em default_path com a ground truth em default_gt (padrão None), cujos caminhos podem ser modificado facilmente. Os resultados dessa operação são armazenados em resultados_binarizacao_nick.
  
  Se o código é > 0,irá processar a imagem desse número da Pasta './Avaliacao/original_images' com a ground truth em './Avaliacao/GT', que são a database do H- DIBCO 2014. Os resultados dessa operação são armazenados em resultado_HDIBCO
  
  Se retornar métricas é verdadeiro e ground truth != None, irá dar print das métricas da imagem, caso contrário não irá.

O outro arquivo de python utilizado é avaliacao.py . Esse arquivo requer a versão 9.4 do Matlab Runtime instalado e ligado na variável ambiental PATH e ira rodar o arquivo oficial de avaliação DIBCO_metrics para avaliar as imagens de 1 a 10 da base de dados. Esse arquivo foi primariamente usado para facilitar a obtenção dos dados para o relatório e, para a maioria dos propósitos, é mais rápido e fácil tomar as métricas geradas por main.py.


