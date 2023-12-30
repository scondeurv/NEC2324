using System.Collections.Immutable;
using TspLibNet;

namespace TSP;

public static class CrossoverExtensions
{
    public static Chromosome OX(this Chromosome @this, Chromosome another, IProblem problem)
    {
        var random = new Random();
        var point1 = random.Next(@this.Genes.Length - 1);
        var point2 = random.Next(@this.Genes.Length - 1);
        var start = Math.Min(point1, point2);
        var end = Math.Max(point1, point2);

        var childGenes = new int[@this.Genes.Length];
        var selectedGenes = @this.Genes.Skip(start).Take(end - start + 1).ToArray();
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

        return new Chromosome(childGenes.ToImmutableArray(), problem);
    }

    public static Chromosome PMX(this Chromosome @this, Chromosome another, IProblem problem)
    {
        var random = new Random();
        var point1 = random.Next(@this.Genes.Length - 1);
        var point2 = random.Next(@this.Genes.Length - 1);
        var start = Math.Min(point1, point2);
        var end = Math.Max(point1, point2);

        var childGenes = new int[@this.Genes.Length];
        var selectedGenes = @this.Genes.Skip(start).Take(end - start + 1).ToArray();

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

        for (var i = end + 1; i < @this.Genes.Length; i++)
        {
            var gene = another.Genes[i];
            while (selectedGenes.Contains(gene))
            {
                var index = Array.IndexOf(selectedGenes, gene);
                gene = another.Genes[start + index];
            }
            childGenes[i] = gene;
        }

        return new Chromosome(childGenes.ToImmutableArray(), problem);
    }
}