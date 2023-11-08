namespace BackPropagation.Configuration;

public record ScalingMethodConfiguration
{
    public required string ScalingMethod { get; init; }
    public (double, double)? Range { get; init; }
}