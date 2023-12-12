using Tools.Common;

namespace KMeans;

public sealed class KMeansClassifier
{
    public async Task<int[]?> Classify(string inputFile, string delimiter, bool noHeader, int k = 2,
        double tolerance = 0.01)
    {
        var dataset = new Dataset();
        await dataset.Load(inputFile, delimiter, noHeader);

        var matrix = dataset
            .ToJagged();
        var input = matrix
            .Select(row => row[0..^1])
            .ToArray();

        var kMeans = new Accord.MachineLearning.KMeans(k);

        kMeans.Tolerance = tolerance;

        var clusters = kMeans.Learn(input);
        var classes = clusters.Decide(input);

        classes = classes.Select(c => c + 1).ToArray();
        return classes;
    }
}