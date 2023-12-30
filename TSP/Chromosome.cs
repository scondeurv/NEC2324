using System.Collections.Immutable;
using TspLibNet;
using TspLibNet.Tours;

namespace TSP;

public class Chromosome
{
    public ImmutableArray<int> Genes { get; init; }
    public double Fitness { get; init; }

    public static Chromosome Create(IProblem problem, Random random)
    {
        var genes = problem.NodeProvider
            .GetNodes()
            .OrderBy(x => random.Next())
            .Select(x => x.Id)
            .ToImmutableArray();
        var fitness = problem.TourDistance(new Tour("tour", string.Empty, genes.Length, genes));
        return new Chromosome(genes, fitness);
    }

    public Chromosome(ImmutableArray<int> genes, IProblem problem)
    {
        Genes = genes;
        Fitness = problem.TourDistance(new Tour("tour", string.Empty, genes.Length, genes));
    }

    private Chromosome(ImmutableArray<int> genes, double fitness)
    {
        Genes = genes;
        Fitness = fitness;
    }
    
    public override string ToString() => @$"
Fitness: {Fitness}
Path: {string.Join(" -> ", Genes.Select(n => n))} -> {Genes.First()}";
}