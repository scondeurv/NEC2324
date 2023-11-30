namespace Tools.Common.Scaling;

public class RobustScaler : IScaler
{
    private double _median;
    private double _iqr;

    public Task<double[]> Scale(double[] data, CancellationToken? cancellationToken = null)
    {
        Array.Sort(data);
        _median = data[data.Length / 2];
        var q1 = data[data.Length / 4];
        var q3 = data[3 * data.Length / 4];
        _iqr = q3 - q1;

        return Task.FromResult(data.Select(x => (x - _median) / _iqr).ToArray());
    }

    public Task<double[]> Descale(double[] scaledData, CancellationToken? cancellationToken = null)
    {
        return Task.FromResult(scaledData.Select(x => x * _iqr + _median).ToArray());
    }
}