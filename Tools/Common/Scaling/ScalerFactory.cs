namespace Tools.Common.Scaling;

public class ScalerFactory
{
    private const string ZScore = "zscore";
    private const string MinMax = "minmax";
    private const string RobustScaler = "robust";
    private const string LogScaler = "log";

    public IReadOnlyDictionary<string, IScaler> CreatePerFeature(
        IReadOnlyDictionary<string, string> configuration)
        => configuration
            .Select(kvp => new { Feature = kvp.Key, ScalingMethod = Create(kvp.Value) })
            .ToDictionary(o => o.Feature, o => o.ScalingMethod);

    private IScaler Create(string method)
        => method switch
        {
            ZScore => new ZCoreScaler(),
            MinMax => new MinMaxScaler(0.0, 1.0),
            RobustScaler => new RobustScaler(),
            LogScaler => new LogScaler(),
            _ => throw new NotSupportedException(method),
        };
}