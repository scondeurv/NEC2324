using CommandLine;

namespace KMeans;

public sealed class Options
{
    [Option(shortName: 'i', longName: "input", Required = true, HelpText = "Input file")]
    public required string InputFile { get; init; }

    [Option(shortName: 's', longName: "separator", Required = false, Default = "\t", HelpText = "Separator")]
    public required string Separator { get; init; }
    
    [Option(shortName: 'h', longName: "no-header", Required = false, Default = false, HelpText = "No Header")]
    public required bool NoHeader { get; init; }
    
    [Option(shortName: 'k', longName: "kvalue", Required = false, Default = 2, HelpText = "KMeans K")]
    public required int K { get; init; }
    
    [Option(shortName: 't', longName: "tolerance", Required = false, Default = 0.01, HelpText = "Tolerance")]
    public required double Tolerance { get; init; }
    
    [Option(shortName: 'd', longName: "distance-method", Required = false, Default = "SquareEuclidean", HelpText = "Distance method: SquareEuclidean, Euclidean, Manhattan, Chebyshev, Minkowski, Canberra, BrayCurtis, Cosine, Jaccard, Dice, Hamming")]
    public required string DistanceMethod { get; init; }
}