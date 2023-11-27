namespace Tools.Common.Scaling;

public class ScalingMethodFactory
{
    private const string ZScore = "zscore";
    private const string MinMax = "minmax";

    public IReadOnlyDictionary<string, IScalingMethod> CreatePerFeature(
        IReadOnlyDictionary<string, string> configuration)
        => configuration
            .Select(kvp => new { Feature = kvp.Key, ScalingMethod = Create(kvp.Value) })
            .ToDictionary(o => o.Feature, o => o.ScalingMethod);

    private IScalingMethod Create(string method)
        => method switch
        {
            ZScore => new ZCore(),
            MinMax => new MinMax(0.000001, 1.0),
            _ => throw new NotSupportedException(method),
        };
}