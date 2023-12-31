using System.Collections.Immutable;
using TspLibNet;
using TspLibNet.Tours;

namespace TSP;

public class Chromosome
{
    public ImmutableArray<int> Genes { get; init; }
    public double Fitness { get; init; }

    public Chromosome(ImmutableArray<int> genes, double fitness)
    {
        Genes = genes;
        Fitness = fitness;
    }
    
    public override string ToString() => @$"
Fitness: {Fitness}
Path: {string.Join(" -> ", Genes.Select(n => n))} -> {Genes.First()}";
}