namespace Tools.Common.Scaling;

public class ZCore : IScalingMethod
{
    private double? Mean { get; set; }
    private double? StdDev { get; set; }

    public async Task<double[]> Scale(double[] data, CancellationToken? cancellationToken)
    {
        if (data is null || data.Length == 0)
        {
            throw new ArgumentNullException(nameof(data));
        }

        Mean = await CalculateMean(data, cancellationToken);
        StdDev = await CalculateStandardDeviation(Mean.Value, data, cancellationToken);
        var scaledData = new double[data.Length];

        for (var i = 0; i < data.Length; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            scaledData[i] = (data[i] - Mean.Value) / StdDev.Value;
        }

        return scaledData;
    }

    public Task<double[]> Descale(double[] scaledData, CancellationToken? cancellationToken = null)
    {
        if (Mean is null || StdDev is null)
        {
            throw new NotSupportedException("Scale must be run first!");
        }

        var descaledData = new double[scaledData.Length];
        for (var i = 0; i < scaledData.Length; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            descaledData[i] = Mean.Value + scaledData[i] * StdDev.Value;
        }

        return Task.FromResult(descaledData);
    }

    private static Task<double> CalculateMean(double[] data, CancellationToken? cancellationToken)
    {
        double acc = 0;
        foreach (var number in data)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            acc += number;
        }

        var mean = acc / data.Length;
        return Task.FromResult(mean);
    }

    private static Task<double> CalculateStandardDeviation(double mean, double[] data,
        CancellationToken? cancellationToken)
    {
        double acc = 0;
        foreach (var value in data)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            acc += Math.Pow(value - mean, 2);
        }

        var standardDeviation = Math.Sqrt(acc / data.Length);
        return Task.FromResult(standardDeviation);
    }
}