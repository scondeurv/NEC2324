using System.Collections.Immutable;
using TspLibNet;

namespace TSP;

public static class MutationExtensions
{
    public static Chromosome RotationToRightMutation(this Chromosome c, IProblem problem)
    {
        var random = new Random();
        var shiftAmount = random.Next(c.Genes.Length);
        var newGenes = new int[c.Genes.Length];

        for (int i = 0; i < c.Genes.Length; i++)
        {
            newGenes[(i + shiftAmount) % c.Genes.Length] = c.Genes[i];
        }

        return new Chromosome(newGenes.ToImmutableArray(), problem);
    }
    
    public static Chromosome InversionMutation(this Chromosome c, IProblem problem)
    {
        var random = new Random();
        var point1 = random.Next(c.Genes.Length);
        var point2 = random.Next(c.Genes.Length);
        var start = Math.Min(point1, point2);
        var end = Math.Max(point1, point2);
        var newGenes = c.Genes.ToArray();
        Array.Reverse(newGenes, start, end - start + 1);
        return new Chromosome(newGenes.ToImmutableArray(), problem);
    }
    
}