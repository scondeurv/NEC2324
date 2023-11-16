namespace BackPropagation.Configuration;

public sealed record NeuralNetworkParameters
{
    public required string TrainingFile { get; init; }
    public string? TestFile { get; init; }
    public required double ValidationPercentage { get; init; }
    public required int Layers { get; init; }
    public required int[] UnitsPerLayer { get; init; }
    public required int Epochs { get; init; }
    public int? BatchSize { get; init; }
    public required float LearningRate { get; init; }
    public required float Momentum { get; init; }
    public required IReadOnlyDictionary<string, ScalingMethodConfiguration> ScalingConfiguration { get; init; }
    public required ActivationFunctionType[] ActivationFunctionPerLayer { get; init; }
    public string? OutputFile { get; init; }
}