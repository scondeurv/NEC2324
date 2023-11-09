namespace BackPropagation.Scaling;

public class ZCore : IScalingMethod
{
    public async Task<double[]> Scale(double[] data, CancellationToken? cancellationToken)
    {
        if (data is null || data.Length == 0)
        {
            throw new ArgumentNullException(nameof(data));
        }
        
        var mean = await CalculateMean(data, cancellationToken);
        var standardDeviation = await CalculateStandardDeviation(mean, data, cancellationToken);
        var scaledData = new double[data.Length];

        for (var i = 0; i < data.Length; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            scaledData[i] = (data[i] - mean) / standardDeviation;
        }
        
        return scaledData;
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
        foreach (var number in data)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            acc += (number - mean) * (number - mean);
        }
        
        var standardDeviation = Math.Sqrt(acc / data.Length);
        return Task.FromResult(standardDeviation);
    }
}