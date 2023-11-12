namespace BackPropagation.Configuration;

public record ScalingMethodConfiguration
{
    public required string Method { get; init; }
    public required double RangeMin { get; init; }
    public required double RangeMax { get; init; }
}