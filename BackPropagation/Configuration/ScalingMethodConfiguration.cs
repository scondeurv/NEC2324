namespace BackPropagation.Configuration;

public record ScalingMethodConfiguration
{
    public string ScalingMethod { get; init; }
    public IReadOnlyDictionary<string, string>? Parameters { get; init; }
}