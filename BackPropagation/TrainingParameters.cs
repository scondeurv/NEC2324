namespace BackPropagation;

public sealed record TrainingParameters
{
    public required string DataFile { get; init; }
    public required int TrainingDataPercentage { get; init; }
    //public required string ActivationFunctionType { get; init; }
    public required int Layers { get; init; }
    public required int UnitPerLayer { get; init; }
    public required int Epochs { get; init; }
    public required float LearningRate { get; init; }
    public required float Momentum { get; init; }
    public required string ScalingMethod { get; init; }
    public string? ScalingMethodParams { get; init; }
    public string? ActivationFunction { get; init; }
    public string? OutputFile { get; init; }
}