import csv
import random
import math
import copy

class Indiv:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = 0.0

def load_data(filename="CaixeiroGruposGA.csv", group_id=3.0):
    points = []
    origin = None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            idx = 1
            for row in reader:
                if len(row) >= 4:
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    grp = float(row[3])
                    if grp == 0.0:
                        origin = {
                            'id': 'Origem',
                            'x': x,
                            'y': y,
                            'z': z
                        }
                    elif grp == group_id:
                        points.append({
                            'id': f"P{idx}",
                            'x': x,
                            'y': y,
                            'z': z
                        })
                        idx += 1
    except FileNotFoundError:
        print(f"Arquivo {filename} não encontrado. Gerando 40 pontos aleatórios para simulação.")
        origin = {'id': 'Origem', 'x': 0.0, 'y': 0.0, 'z': 0.0}
        for i in range(1, 41):
            points.append({
                'id': f"P{i}",
                'x': random.uniform(0, 100),
                'y': random.uniform(0, 100),
                'z': random.uniform(0, 100)
            })
            
    if origin is None:
        origin = {'id': 'Origem', 'x': 0.0, 'y': 0.0, 'z': 0.0}
        
    return [origin] + points


def calc_distance(p1, p2):
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)

class TSPGeneticAlgorithm:
    def __init__(self, points, pop_size=100, max_gen=500, mutation_rate=0.01, elitism_size=2):
        self.points = points
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size
        
        self.num_genes = len(points) - 1
        self.intermediate_indices = list(range(1, len(points)))
        
    def initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            chrom = copy.deepcopy(self.intermediate_indices)
            random.shuffle(chrom)
            pop.append(Indiv(chrom))
        return pop
        
    def evaluate(self, individual):
        dist = 0.0
        dist += calc_distance(self.points[0], self.points[individual.chromosome[0]])
        
        for i in range(len(individual.chromosome) - 1):
            p1_idx = individual.chromosome[i]
            p2_idx = individual.chromosome[i+1]
            dist += calc_distance(self.points[p1_idx], self.points[p2_idx])
            
        dist += calc_distance(self.points[individual.chromosome[-1]], self.points[0])
        
        individual.fitness = dist
        
    def tournament_selection(self, pop, k=3):
        competitors = random.sample(pop, k)
        best = min(competitors, key=lambda ind: ind.fitness)
        return best
        
    def crossover_order(self, p1, p2):
        c1, c2 = sorted(random.sample(range(self.num_genes), 2))
        
        offspring_chrom = [-1] * self.num_genes
        offspring_chrom[c1:c2+1] = p1.chromosome[c1:c2+1]
        
        p2_idx = 0
        for i in range(self.num_genes):
            if offspring_chrom[i] == -1:
                while p2.chromosome[p2_idx] in offspring_chrom:
                    p2_idx += 1
                offspring_chrom[i] = p2.chromosome[p2_idx]
                
        return Indiv(offspring_chrom)

    def mutate_swap(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.num_genes), 2)
            individual.chromosome[idx1], individual.chromosome[idx2] = individual.chromosome[idx2], individual.chromosome[idx1]

    def run(self):
        pop = self.initialize_population()
        for ind in pop:
            self.evaluate(ind)
            
        best_overall = min(pop, key=lambda ind: ind.fitness)
        history = [best_overall.fitness]
        
        gen = 0
        stagnation = 0
        
        while gen < self.max_gen:
            pop.sort(key=lambda ind: ind.fitness)
            new_pop = []
            
            if self.elitism_size > 0:
                new_pop.extend(copy.deepcopy(pop[:self.elitism_size]))
                
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_selection(pop)
                p2 = self.tournament_selection(pop)
                
                offspring = self.crossover_order(p1, p2)
                self.mutate_swap(offspring)
                
                self.evaluate(offspring)
                new_pop.append(offspring)
                
            pop = new_pop
            best_current = min(pop, key=lambda ind: ind.fitness)
            history.append(best_current.fitness)
            
            if best_current.fitness < best_overall.fitness:
                best_overall = copy.deepcopy(best_current)
                stagnation = 0
            else:
                stagnation += 1
                
            if stagnation >= 50:
                break
                
            gen += 1
            
        return best_overall, gen, history

def analyse_frequentist(points, runs=30):
    print(f"\n--- Iniciando Análise Frequentista ({runs} rodadas) ---")
    generations_needed = []
    
    for i in range(runs):
        ag = TSPGeneticAlgorithm(points, pop_size=100, max_gen=300, mutation_rate=0.01, elitism_size=2)
        best, gen, hist = ag.run()
        generations_needed.append(gen)
        if (i+1) % 5 == 0:
            print(f"Rodada {i+1}/{runs} concluída.")
            
    from collections import Counter
    counts = Counter(generations_needed)
    moda = counts.most_common(1)[0][0]
    
    print("\nResultados da Análise Frequentista:")
    print(f"Moda das gerações para convergência/estagnação: {moda}")
    print(f"Média das gerações: {sum(generations_needed)/runs:.2f}")

if __name__ == "__main__":
    pts = load_data()
    
    print(f"Total de pontos carregados: {len(pts)}")
    
    ag = TSPGeneticAlgorithm(pts, pop_size=100, max_gen=500, mutation_rate=0.01, elitism_size=2)
    best_ind, gens, history = ag.run()
    
    route_ids = [pts[0]['id']] + [pts[i]['id'] for i in best_ind.chromosome] + [pts[0]['id']]
    route_str = " -> ".join(route_ids)
    
    print(f"\nMelhor rota encontrada em {gens} gerações:")
    print(f"[{route_str}]")
    print(f"Custo (Distância Euclideana 3D total): {best_ind.fitness:.4f}")
    
    analyse_frequentist(pts, runs=15)
