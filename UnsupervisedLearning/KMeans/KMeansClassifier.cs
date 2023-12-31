﻿using Accord.Math.Distances;
using Tools.Common;

namespace KMeans;

public sealed class KMeansClassifier
{
    public async Task<int[]?> Classify(string inputFile, string delimiter, bool noHeader, int k = 2,
        double tolerance = 0.01, string distanceMethod = "SquareEuclidean")
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
        kMeans.Distance = GetDistanceMethod(distanceMethod);

        var clusters = kMeans.Learn(input);
        var predictedClasses = clusters.Decide(input);

        predictedClasses = predictedClasses.Select(c => c + 1).ToArray();
        return predictedClasses;
    }

    private static IDistance<double[], double[]> GetDistanceMethod(string distanceMethod)
        => distanceMethod.ToLower() switch
        {
            "squareeuclidean" => new SquareEuclidean(),
            "euclidean" => new Euclidean(),
            "manhattan" => new Manhattan(),
            "chebyshev" => new Chebyshev(),
            "minkowski" => new Minkowski(),
            "canberra" => new Canberra(),
            "braycurtis" => new BrayCurtis(),
            "cosine" => new Cosine(),
            "jaccard" => new Jaccard(),
            "dice" => new Dice(),
            "hamming" => new Hamming(),
            _ => throw new NotSupportedException(distanceMethod)
        };
}