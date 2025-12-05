import random
import numpy as np
from typing import List

class InfluenzaReassortant:
    """Represents an influenza A virus reassortant with 8 segments"""
    
    # Segment indices
    PB2, PB1, PA, HA, NP, NA, M, NS = 0, 1, 2, 3, 4, 5, 6, 7
    SEGMENT_NAMES = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M', 'NS']
    
    def __init__(self, genome: List[int] = None):
        if genome is None:
            self.genome = [random.randint(0, 1) for _ in range(8)]
        else:
            self.genome = genome[:]
        self.fitness = 0.0
    
    def __str__(self):
        segments = [f"{self.SEGMENT_NAMES[i]}:P{self.genome[i]}" for i in range(8)]
        return f"[{', '.join(segments)}] Fitness: {self.fitness:.1f}"
    
    def copy(self):
        new_reassortant = InfluenzaReassortant(self.genome)
        new_reassortant.fitness = self.fitness
        return new_reassortant

class InfluenzaGA:
    """Genetic Algorithm for optimizing influenza A virus reassortants"""
    
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def fitness_function(self, reassortant: InfluenzaReassortant) -> float:
        """Calculate fitness based on functional segment relationships"""
        genome = reassortant.genome
        fitness = 0.0
        
        # 1. Polymerase complex (PB2, PB1, PA) from same parent
        polymerase_segments = [genome[0], genome[1], genome[2]]
        if len(set(polymerase_segments)) == 1:
            fitness += 100  # High score for intact polymerase complex
            polymerase_parent = polymerase_segments[0]
            
            # 2. NP associated with polymerase complex
            if genome[4] == polymerase_parent:
                fitness += 50
        else:
            # Partial polymerase integrity
            if polymerase_segments.count(0) == 2 or polymerase_segments.count(1) == 2:
                fitness += 35
        
        # 3. HA-NA pairing
        if genome[3] == genome[5]:
            fitness += 40
        
        # Small penalty for complete parental types (encourage reassortment)
        if genome.count(0) == 8 or genome.count(1) == 8:
            fitness -= 70
        
        return fitness
    
    def initialize_population(self):
        """Initialize random population"""
        self.population = []
        for _ in range(self.population_size):
            reassortant = InfluenzaReassortant()
            reassortant.fitness = self.fitness_function(reassortant)
            self.population.append(reassortant)
    
    def selection(self) -> InfluenzaReassortant:
        """Tournament selection"""
        tournament = random.sample(self.population, 3)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: InfluenzaReassortant, parent2: InfluenzaReassortant):
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, 7)
        child1_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
        child2_genome = parent2.genome[:crossover_point] + parent1.genome[crossover_point:]
        
        return InfluenzaReassortant(child1_genome), InfluenzaReassortant(child2_genome)
    
    def mutate(self, reassortant: InfluenzaReassortant):
        """Flip mutation"""
        for i in range(8):
            if random.random() < self.mutation_rate:
                reassortant.genome[i] = 1 - reassortant.genome[i]
    
    def evolve_generation(self):
        """Evolve one generation"""
        new_population = []
        
        # Elitism - keep best individual
        best_individual = max(self.population, key=lambda x: x.fitness)
        new_population.append(best_individual.copy())
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            child1.fitness = self.fitness_function(child1)
            child2.fitness = self.fitness_function(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def run(self, generations=100):
        """Run the genetic algorithm"""
        print("Starting GA optimization...")
        self.initialize_population()
        
        for gen in range(generations):
            self.evolve_generation()
            
            # Track statistics
            fitnesses = [ind.fitness for ind in self.population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            if gen % 20 == 0 or gen == generations - 1:
                print(f"Generation {gen}: Best={best_fitness:.1f}, Avg={avg_fitness:.1f}")
        
        print(f"Optimization completed after {generations} generations.")
    
    def get_top_reassortants(self, n=200) -> List[InfluenzaReassortant]:
        """Get top n reassortants"""
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n]
    
    def analyze_results(self):
        """Analyze and display results"""
        top_200 = self.get_top_reassortants(200)
        
        print(f"\n{'='*60}")
        print("TOP 200 REASSORTANT COMBINATIONS")
        print(f"{'='*60}")
        
        for i, reassortant in enumerate(top_200, 1):
            print(f"\n{i}. {reassortant}")
            
            # Analyze polymerase complex
            pol_segments = [reassortant.genome[0], reassortant.genome[1], reassortant.genome[2]]
            if len(set(pol_segments)) == 1:
                pol_parent = "Parent " + str(pol_segments[0])
                np_parent = "Parent " + str(reassortant.genome[4])
                np_match = "✓" if reassortant.genome[4] == pol_segments[0] else "✗"
                print(f"   Polymerase complex: {pol_parent} (intact)")
                print(f"   NP: {np_parent} {np_match}")
            else:
                print(f"   Polymerase complex: Mixed {pol_segments} (not intact)")
            
            # Analyze HA-NA pairing
            ha_parent = "Parent " + str(reassortant.genome[3])
            na_parent = "Parent " + str(reassortant.genome[5])
            ha_na_match = "✓" if reassortant.genome[3] == reassortant.genome[5] else "✗"
            print(f"   HA-NA pair: {ha_parent}, {na_parent} {ha_na_match}")
            
            # Count segments from each parent
            from_p0 = reassortant.genome.count(0)
            from_p1 = reassortant.genome.count(1)
            print(f"   Segments from Parent 0: {from_p0}, Parent 1: {from_p1}")

# Simple usage
if __name__ == "__main__":
    print("Influenza A Reassortant Optimization - Proof of Concept")
    print("="*60)
    
    # Run GA
    ga = InfluenzaGA(population_size=50, mutation_rate=0.1, crossover_rate=0.8)
    ga.run(generations=100)
    
    # Show results
    ga.analyze_results()
    
    # Show convergence
    print(f"\nConvergence: Best fitness improved from {ga.best_fitness_history[0]:.1f} to {ga.best_fitness_history[-1]:.1f}")
