using System.Collections.Immutable;
using Accord.MachineLearning;
using Accord.Math.Distances;
using Accord.Statistics.Distributions.Multivariate;
using Tools.Common;

namespace KMeans;

public sealed class KMeansClassifier
{
    public async Task<int[]?> Classify(string inputFile, string delimiter, bool noHeader, int k = 2)
    {
        var dataset = new Dataset();
        await dataset.Load(inputFile, delimiter, noHeader);
        
        var matrix = dataset
            .ToJagged();
        var input = matrix
            .Select(row => row[0..^1])
            .ToArray();
        
        var kMeans = new Accord.MachineLearning.KMeans(k)
        {
            ParallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = 5,
            },
            Distance = new SquareEuclidean(),
            Tolerance = 0.00001,
            MaxIterations = 10000,
            UseSeeding = Seeding.KMeansPlusPlus,
        };
        
        var clusters = kMeans.Learn(input);
        var classes = clusters.Decide(input);

        return classes;
    }
}