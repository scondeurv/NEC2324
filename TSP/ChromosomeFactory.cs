using System.Collections.Immutable;
using TspLibNet;
using TspLibNet.Tours;

namespace TSP;

public static class ChromosomeFactory
{
    private static Random Random { get; } = new();
    
    public static Chromosome Create(IProblem problem)
    {
        var genes = problem.NodeProvider
            .GetNodes()
            .OrderBy(x => Random.Next())
            .Select(x => x.Id)
            .ToImmutableArray();

        var fitness = 1 / problem.TourDistance(new Tour("tour", string.Empty, genes.Length, genes));
        return new Chromosome(genes, fitness);
    }
    
    public static Chromosome Create (ImmutableArray<int> genes, IProblem problem)
    {
        var fitness = 1 / problem.TourDistance(new Tour("tour", string.Empty, genes.Length, genes));
        return new Chromosome(genes, fitness);
    }
}