namespace BackPropagation.Configuration;

public record ScalingMethodConfiguration
{
    public required string Method { get; init; }
    public (double, double)? Range { get; init; }
}