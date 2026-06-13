import random
import math
import copy

# Função de Rastrigin
def rastrigin(x):
    n = len(x)
    return 10 * n + sum((xi**2 - 10 * math.cos(math.pi * xi)) for xi in x)

class IndivReal:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = 0.0

class GAReal:
    def __init__(self, dim=50, pop_size=100, max_gen=500, bounds=(-5.12, 5.12)):
        self.dim = dim
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.bounds = bounds
        
    def init_pop(self):
        pop = []
        for _ in range(self.pop_size):
            chrom = [random.uniform(self.bounds[0], self.bounds[1]) for _ in range(self.dim)]
            pop.append(IndivReal(chrom))
        return pop
        
    def evaluate(self, ind):
        ind.fitness = rastrigin(ind.chromosome) # Minimização
        
    def tournament(self, pop, k=3):
        competitors = random.sample(pop, k)
        return min(competitors, key=lambda ind: ind.fitness)
        
    def sbx_crossover(self, p1, p2, eta=1.0):
        # Simulated Binary Crossover (SBX) com eta = 1
        c1 = []
        c2 = []
        for x1, x2 in zip(p1.chromosome, p2.chromosome):
            u = random.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
            off1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            off2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            
            # Limita ao intervalo (bounds)
            off1 = max(self.bounds[0], min(self.bounds[1], off1))
            off2 = max(self.bounds[0], min(self.bounds[1], off2))
            
            c1.append(off1)
            c2.append(off2)
            
        return IndivReal(c1), IndivReal(c2)
        
    def gaussian_mutation(self, ind, prob=0.1, sigma=0.5):
        # Mutação Genética Gaussiana
        for i in range(self.dim):
            if random.random() < prob:
                ind.chromosome[i] += random.gauss(0, sigma)
                ind.chromosome[i] = max(self.bounds[0], min(self.bounds[1], ind.chromosome[i]))
                
    def run(self):
        pop = self.init_pop()
        for ind in pop:
            self.evaluate(ind)
            
        best = min(pop, key=lambda x: x.fitness)
        
        for gen in range(self.max_gen):
            new_pop = [copy.deepcopy(best)] # Elitismo (preserva o melhor)
            
            while len(new_pop) < self.pop_size:
                p1 = self.tournament(pop)
                p2 = self.tournament(pop)
                
                # Recombinação SBX
                o1, o2 = self.sbx_crossover(p1, p2, eta=1.0)
                
                # Mutação Gaussiana
                self.gaussian_mutation(o1, prob=1.0/self.dim)
                self.gaussian_mutation(o2, prob=1.0/self.dim)
                
                self.evaluate(o1)
                self.evaluate(o2)
                
                new_pop.extend([o1, o2])
                
            pop = new_pop[:self.pop_size]
            current_best = min(pop, key=lambda x: x.fitness)
            if current_best.fitness < best.fitness:
                best = copy.deepcopy(current_best)
                
        # Retorna o custo computacional em número de avaliações
        cost = self.pop_size * self.max_gen
        return best.fitness, cost

def hill_climbing(dim=50, max_iter=50000, bounds=(-5.12, 5.12), step_size=0.1):
    # Algoritmo de Busca Local para confronto
    current_state = [random.uniform(bounds[0], bounds[1]) for _ in range(dim)]
    current_fit = rastrigin(current_state)
    
    for _ in range(max_iter):
        neighbor = copy.deepcopy(current_state)
        idx = random.randint(0, dim - 1)
        neighbor[idx] += random.uniform(-step_size, step_size)
        neighbor[idx] = max(bounds[0], min(bounds[1], neighbor[idx]))
        
        neighbor_fit = rastrigin(neighbor)
        
        if neighbor_fit < current_fit: # Minimização
            current_state = neighbor
            current_fit = neighbor_fit
            
    cost = max_iter
    return current_fit, cost

def comparar_metodos():
    dimensao = 50
    print(f"--- Otimização Função Rastrigin (Dimensão = {dimensao}) ---")
    
    # 1. Base de Comparação: Hill Climbing
    print("\n[1] Executando Hill Climbing (Base de Comparação)...")
    hc_max_iter = 50000
    hc_fit, hc_cost = hill_climbing(dim=dimensao, max_iter=hc_max_iter)
    print(f"Melhor fitness (Hill Climbing): {hc_fit:.4f} | Custo Computacional: {hc_cost} avaliações")
    
    # 2. Algoritmo Genético Não-Canônico (Variação de População)
    pop_sizes = [20, 50, 100, 200]
    
    print("\n[2] Executando AG Não-Canônico (Análise Custo x Convergência)...")
    for pop in pop_sizes:
        # Ajusta max_gen para tentar manter o custo máximo em 50.000 (para equivalência) 
        # ou fixar gen para ver as diferentes qualidades de convergência
        max_gen = 50000 // pop
        ag = GAReal(dim=dimensao, pop_size=pop, max_gen=max_gen)
        best_fit, ag_cost = ag.run()
        print(f"AG População: {pop:3d} | Max Gerações: {max_gen:4d} | Melhor fitness: {best_fit:.4f} | Custo Computacional: {ag_cost} avaliações")

if __name__ == "__main__":
    comparar_metodos()
