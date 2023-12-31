using System.Collections.Immutable;
using TspLibNet;

namespace TSP;

public static class Crossover
{
    private static Random Random { get; } = new();
    public static Chromosome DoCrossover(string method, Chromosome one, Chromosome another, IProblem problem)
    {
        switch (method.ToLower())
        {
            case "ox":
                return OX(one, another, problem);
            case "pmx":
                return PMX(one, another, problem);
            default:
                throw new NotSupportedException(method);
        }
    }
    
    public static Chromosome OX(Chromosome one, Chromosome another, IProblem problem)
    {
        var point1 = Random.Next(one.Genes.Length - 1);
        var point2 = Random.Next(one.Genes.Length - 1);
        var start = Math.Min(point1, point2);
        var end = Math.Max(point1, point2);

        var childGenes = new int[one.Genes.Length];
        var selectedGenes = one.Genes.Skip(start).Take(end - start + 1).ToArray();
        var remainingGenes = another.Genes.Where(g => !selectedGenes.Contains(g));

        for (var i = 0; i < childGenes.Length; i++)
        {
            if (i >= start && i <= end)
            {
                childGenes[i] = selectedGenes[i - start];
                continue;
            }

            childGenes[i] = remainingGenes.First();
            remainingGenes = remainingGenes.Skip(1);
        }

        return ChromosomeFactory.Create(childGenes.ToImmutableArray(), problem);
    }

    public static Chromosome PMX(Chromosome one, Chromosome another, IProblem problem)
    {
        var point1 = Random.Next(one.Genes.Length - 1);
        var point2 = Random.Next(one.Genes.Length - 1);
        var start = Math.Min(point1, point2);
        var end = Math.Max(point1, point2);

        var childGenes = new int[one.Genes.Length];
        var selectedGenes = one.Genes.Skip(start).Take(end - start + 1).ToArray();

        for (var i = start; i <= end; i++)
        {
            childGenes[i] = selectedGenes[i - start];
        }

        for (var i = 0; i < start; i++)
        {
            var gene = another.Genes[i];
            while (selectedGenes.Contains(gene))
            {
                var index = Array.IndexOf(selectedGenes, gene);
                gene = another.Genes[start + index];
            }
            childGenes[i] = gene;
        }

        for (var i = end + 1; i < one.Genes.Length; i++)
        {
            var gene = another.Genes[i];
            while (selectedGenes.Contains(gene))
            {
                var index = Array.IndexOf(selectedGenes, gene);
                gene = another.Genes[start + index];
            }
            childGenes[i] = gene;
        }

        return ChromosomeFactory.Create(childGenes.ToImmutableArray(), problem);
    }
}