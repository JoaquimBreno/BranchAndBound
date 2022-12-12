
import argparse
from mip import *
import numpy as np
import os

VALUE_SEGMENTATION = 0.5

class BranchAndBound(object):
  def __init__(self, file_name):
    # ## ENTRADA
    self.file_name = file_name
    self.num_vars = 0
    self.num_rests = 0
    self.objective_function = []
    self.restrictions = []
    # ## AUXILIARES
    self.primal = 0
    self.optimal_model = None
  
  #Função que lê arquivo TXT e seta as variavéis globais
  def read_txt(self):  #CAMINHO DO ARQUVIVO EM STRING, EXEMPLO: 'projeto.txt'

    with open(self.file_name) as file:
      lines = [line.split() for line in file]

    self.num_vars = int(lines[0][0])
    self.num_rests = int(lines[0][1])

    for i in range(1, len(lines)):
      if i == 1:
        self.objective_function = [int(x) for x in lines[i]]
      else:
        self.restrictions.append([int(x) for x in lines[i]])
        
  def create_model(self): #Função que cria o modelo MIP e retorna o modelo

    model = Model(sense=MAXIMIZE) # Cria modelo MIP
    x = [model.add_var(var_type="CONTINUOUS",  # Adiciona variável contínua
                        lb=0, ub=1, name="x_" + str(i)) for i in range(self.num_vars)]

    model.objective = xsum(x[i]*self.objective_function[i] for i in range(self.num_vars)) # Cria função objetivo

    # Cria restrições
    for i in range(self.num_rests): # Itera o número de restrições
      model += xsum(self.restrictions[i][j]*x[j] for j in range(self.num_vars)) <= self.restrictions[i][-1] # Adiciona variáveis e coeficientes da restrição e o lado direito da restrição <= 

    return model

  def solver(self, model): #Função que recebe um modelo e retorna um dicionário com o valor da função objetivo e uma lista com as variáveis do modelo
    model.verbose = 0
    model.optimize()
    params = {}
    params["objective"] = model.objective_value
    params["vars"] = model.vars

    return params

  def branch_and_bound(self,model): #Função branch and bound, ramifica e poda o modelo, tem como entrada o modelo inicial e não retorna nada, apenas atualiza o primal e o modelo ótimo
    nodes = [model]  #Fila

    while len(nodes): 
      flag, solving = self.bound(nodes[0]) # Verifica se o modelo é inviável, limitante, fracionário ou integral, podando o modelo de acordo com a flag != 'FRACIONÁRIO'
      if flag in ['INVIABILIDADE','LIMITADA']: # Poda por descontinuação, seja por ser inviável ou limitada.
        nodes.pop(0) 
      elif flag == 'INTEGRALIDADE': # Se for integral, atualizamos o primal e o modelo ótimo, e remove-se o modelo atual da lista
        if solving["objective"] > self.primal:
          self.optimal_model = nodes[0]
          self.primal = solving["objective"]
        nodes.pop(0)
      elif flag == 'FRACIONÁRIO':   #Se for fracionário, ramifica-se o modelo atual e adiciona-se os modelos ramificados na lista
        flag_branch = self.branch(nodes[0], solving["vars"])
        nodes.append(flag_branch[0])
        nodes.append(flag_branch[1])
        nodes.pop(0)

  def branch(self, model, vars_solved): # Função branch, faz a ramificação do modelo adotando as restrições 0 e 1 pra a variável encontrada mais próxima do VALUE_SEGMENTATION
    var_branch = vars_solved[self.find_value([i.x for i in vars_solved], VALUE_SEGMENTATION)] # Aqui seleciona-se a variável de valor fracionário mais próximo do valor de segmentação (0.5)

    model_0 = model.copy()   #Restrição var == 0
    model_0 += var_branch == 0
    model_1 = model.copy()   #Restrição var == 1
    model_1 += var_branch == 1

    return (model_0, model_1)

  def bound(self, model): # Função bound, verifica se o modelo resolvido é inviável, limitante, fracionário ou integral

    solved = self.solver(model)
    fractional = False

    if solved["objective"] == None: 
      return 'INVIABILIDADE', None

    for i in range(len(solved["vars"])): # Se o valor de alguma variável for fracionário, retorna fracionário
      if solved["vars"][i].x == int(solved["vars"][i].x):
        continue
      else:
        fractional = True
        break
    
    if fractional == False: # Se não for fracionário, retorna integralidade
      print("Encontrada integralidade")
      return 'INTEGRALIDADE', solved

    if solved["objective"] <= self.primal: 
      return 'LIMITADA', None

    return 'FRACIONÁRIO', solved    # Se não for nem fracionário nem limitada, retorna fracionário, para ser ramificado

  def find_value(self, array, value):   #Função que recebe uma lista de variável e um valor, busca na lista a variável mais próxima do valor e retorna o valor da variavel
      array = np.asarray(array)
      value_found = np.absolute(array - value)
      value_found = value_found.argmin()
      return value_found

  def main(self):
    self.read_txt()

    start_model = self.create_model()

    self.branch_and_bound(start_model)
    solution = self.solver(self.optimal_model)

    print("Resultados")
    for i in solution["vars"]:
      print(i.name, '=', i.x)
    print("Função Objetivo:")
    print('Z =', solution["objective"])


if __name__ == '__main__':
  # Recebe o nome do arquivo de entrada pelo terminal com o comando python3 branch_and_bound.py --file_name <nome_do_arquivo>
  parser = argparse.ArgumentParser()
  parser.add_argument("--file_name", help="Nome do arquivo de entrada")
  args = parser.parse_args()
  file_name = args.file_name

  # Verifica se o arquivo é .txt
  if file_name[-4:] != ".txt":
    print("Arquivo inválido")
    exit()

  # Verifica se o arquivo existe
  if not os.path.isfile(file_name):
    print("Arquivo não encontrado")
    exit()

  print("Projeto Branch and Bound - Joaquim Breno")
  bab=BranchAndBound(file_name)
  bab.main()