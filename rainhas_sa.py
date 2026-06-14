import random
import math
import copy

class NQueensSA:
    def __init__(self, n=8):
        self.n = n
        self.max_fitness = 28  # Para 8 rainhas: (8 * 7) / 2
        
    def initial_state(self):
        state = list(range(self.n))
        random.shuffle(state)
        return state
        
    def fitness(self, state):
        h = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                    h += 1
        return self.max_fitness - h
        
    def get_neighbor(self, state):
        neighbor = copy.deepcopy(state)
        col = random.randint(0, self.n - 1)
        new_row = random.randint(0, self.n - 1)
        while new_row == state[col]:
            new_row = random.randint(0, self.n - 1)
        neighbor[col] = new_row
        return neighbor

    def simulated_annealing(self, initial_temp=100.0, cooling_rate=0.99, min_temp=0.01, max_iter=10000):
        current_state = self.initial_state()
        current_fit = self.fitness(current_state)
        
        best_state = current_state
        best_fit = current_fit
        
        temp = initial_temp
        iterations = 0
        
        while temp > min_temp and iterations < max_iter:
            if best_fit == self.max_fitness:
                break
                
            neighbor = self.get_neighbor(current_state)
            neighbor_fit = self.fitness(neighbor)
            
            delta_fit = neighbor_fit - current_fit
            
            if delta_fit > 0 or random.random() < math.exp(delta_fit / temp):
                current_state = neighbor
                current_fit = neighbor_fit
                
                if current_fit > best_fit:
                    best_state = current_state
                    best_fit = current_fit
            
            temp *= cooling_rate
            iterations += 1
            
        return best_state, best_fit, iterations

def buscar_92_solucoes():
    solver = NQueensSA()
    solutions = set()
    total_iterations = 0
    runs = 0
    
    print("Iniciando busca pelas 92 soluções...")
    
    while len(solutions) < 92:
        runs += 1
        state, fit, iters = solver.simulated_annealing(initial_temp=100, cooling_rate=0.95, min_temp=0.001, max_iter=5000)
        total_iterations += iters
        
        if fit == 28:
            solution_tuple = tuple(state)
            if solution_tuple not in solutions:
                solutions.add(solution_tuple)
                if len(solutions) % 10 == 0 or len(solutions) == 92:
                    print(f"Encontradas {len(solutions)}/92 soluções únicas. Execuções: {runs}, Iterações totais: {total_iterations}")
                    
    print("\nResumo da Busca:")
    print(f"Total de execuções do SA: {runs}")
    print(f"Custo computacional (Iterações totais acumuladas): {total_iterations}")

if __name__ == "__main__":
    buscar_92_solucoes()
