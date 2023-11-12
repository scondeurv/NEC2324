using BackPropagation.Configuration;

namespace BackPropagation.Scaling;

public class ScalingMethodFactory
{
    private const string ZScore = "zscore";
    private const string MinMax = "minmax";

    public IReadOnlyDictionary<string, IScalingMethod> CreatePerFeature(
        IReadOnlyDictionary<string, ScalingMethodConfiguration> configuration)
        => configuration
            .Select(kvp => new { Feature = kvp.Key, SalingMethod = Create(kvp.Value) })
            .ToDictionary(o => o.Feature, o => o.SalingMethod);

    private IScalingMethod Create(ScalingMethodConfiguration configuration)
        => configuration.Method switch
        {
            ZScore => new ZCore(),
            MinMax => new MinMax(configuration.RangeMin, configuration.RangeMax),
            _ => throw new NotSupportedException(configuration.Method),
        };
}