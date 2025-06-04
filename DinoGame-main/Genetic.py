#import random
import numpy as np

def updateNetwork(population):
    # ===================== ESTA FUNCIÓN RECIBE UNA POBLACIÓN A LA QUE SE DEBEN APLICAR MECANISMOS DE SELECCIÓN, =================
    # ===================== CRUCE Y MUTACIÓN. LA ACTUALIZACIÓN DE LA POBLACIÓN SE APLICA EN LA MISMA VARIABLE ====================
    '''
    Actualiza la población aplicando selección, cruce, mutación y elitismo.
    '''
    # Después de obtener los mejores y copiarlos al inicio
    elitismo_num = 3
    mejores = sorted(population, key=lambda d: d.score, reverse=True)[:elitismo_num]

    for i in range(elitismo_num):
        population[i].W1 = mejores[i].W1.copy()
        population[i].b1 = mejores[i].b1.copy()
        population[i].W2 = mejores[i].W2.copy()
        population[i].b2 = mejores[i].b2.copy()
        population[i].W3 = mejores[i].W3.copy()   # AGREGADO
        population[i].b3 = mejores[i].b3.copy() 
        population[i].resetStatus()

    # Luego generas parejas y aplicas cruzas sólo a partir de elitismo_num
    parejas = select_fittest(population, top_n=6)  # Se cambia la cnatidad de padres

    # Nueva población temporal (si quieres mantenerlo puedes crear una lista)
    nuevos_individuos = []

    # Iterar sólo sobre la población desde elitismo_num para reemplazar
    indice_reemplazo = elitismo_num

    for idx1, idx2 in parejas:
        padre1 = population[idx1]
        padre2 = population[idx2]

        hijo1_weights, hijo2_weights = evolve(padre1, padre2)

        for pesos in [hijo1_weights, hijo2_weights]:
            if indice_reemplazo < len(population):
                population[indice_reemplazo].W1 = pesos['W1']
                population[indice_reemplazo].b1 = pesos['b1']
                population[indice_reemplazo].W2 = pesos['W2']
                population[indice_reemplazo].b2 = pesos['b2']
                population[indice_reemplazo].W3 = pesos['W3']   # AGREGADO
                population[indice_reemplazo].b3 = pesos['b3'] 
                population[indice_reemplazo].resetStatus()
                indice_reemplazo += 1


def select_fittest(population, top_n=6): #top_n cantidad de padres que elijo entre los mejores
    """
    Selecciona los top_n mejores individuos según su score para reproducirse.
    Empareja secuencialmente para generar parejas.
    Si top_n es impar, empareja el último con el primero.
    """
    # Ordenar por score descendente
    sorted_indices = sorted(range(len(population)), key=lambda i: population[i].score, reverse=True)

    # Tomar solo los mejores top_n
    selected = sorted_indices[:top_n]

    # Emparejar secuencialmente
    parejas = []
    for i in range(0, len(selected), 2):
        if i+1 < len(selected):
            parejas.append((selected[i], selected[i+1]))
        else:
            parejas.append((selected[i], selected[0]))
    return parejas


def crossover(parent1_matrix, parent2_matrix):
    """
    Crossover uniforme: por cada gen, se elige aleatoriamente cuál padre hereda.
    """
    mask = np.random.rand(*parent1_matrix.shape) < 0.5
    child = np.where(mask, parent1_matrix, parent2_matrix)
    return child

def mutate(matrix, mutation_rate=0.2, mutation_strength=0.03):
    """
    Mutación gaussiana con probabilidad mutation_rate y magnitud mutation_strength.
    """
    mutation_mask = np.random.rand(*matrix.shape) < mutation_rate
    noise = np.random.randn(*matrix.shape) * mutation_strength
    matrix[mutation_mask] += noise[mutation_mask]
    return matrix

def evolve(parent1, parent2):
    # ===================== FUNCIÓN DE CRUCE Y MUTACIÓN =====================
    """
    Cruza y muta los pesos y biases de dos padres, retorna pesos de dos hijos.
    """
    child1 = {}
    child2 = {}

    for attr in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']: 
        p1_matrix = getattr(parent1, attr)
        p2_matrix = getattr(parent2, attr)

        c1_matrix = crossover(p1_matrix, p2_matrix)
        c2_matrix = crossover(p2_matrix, p1_matrix)

        c1_matrix = mutate(c1_matrix)
        c2_matrix = mutate(c2_matrix)

        child1[attr] = c1_matrix
        child2[attr] = c2_matrix

    return child1, child2

    