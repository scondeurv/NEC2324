using System.Collections.Immutable;
using TspLibNet;

namespace TSP;

public static class Mutator
{
    private static Random Random { get; } = new();
    
    public static Chromosome DoMutation(string method, Chromosome c, IProblem problem)
    {
        switch (method.ToLower())
        {
            case "right-shift":
                return RightShiftMutation(c, problem);
            case "inversion":
                return InversionMutation(c, problem);
            default:
                throw new NotSupportedException(method);
        }
    }
    
    public static Chromosome RightShiftMutation(Chromosome c, IProblem problem)
    {
        int shiftAmount;
        lock (Random)
        {
            shiftAmount = Random.Next(c.Genes.Length);
        } 

        var newGenes = new int[c.Genes.Length];

        for (var i = 0; i < c.Genes.Length; i++)
        {
            newGenes[(i + shiftAmount) % c.Genes.Length] = c.Genes[i];
        }

        return ChromosomeFactory.Create(newGenes.ToImmutableArray(), problem);
    }
    
    public static Chromosome InversionMutation(Chromosome c, IProblem problem)
    {
        int point1;
        int point2;
        lock (Random)
        {
            point1 = Random.Next(c.Genes.Length);
            point2 = Random.Next(c.Genes.Length);
        }

        var start = Math.Min(point1, point2);
        var end = Math.Max(point1, point2);
        var newGenes = c.Genes.ToArray();
        Array.Reverse(newGenes, start, end - start + 1);
        return ChromosomeFactory.Create(newGenes.ToImmutableArray(), problem);
    }
}