namespace BackPropagation.Scaling;

public class ZCore : IScalingMethod
{
    public async Task<double[]> Scale(double[] data)
    {
        if (data is null || data.Length == 0)
        {
            throw new ArgumentNullException(nameof(data));
        }
        
        var mean = await CalculateMean(data);
        var standardDeviation = await CalculateStandardDeviation(mean, data);
        var scaledData = new double[data.Length];

        for (var i = 0; i < data.Length; i++)
        {
            scaledData[i] = (data[i] - mean) / standardDeviation;
        }
        
        return scaledData;
    }

    private static Task<double>  CalculateMean(double[] data)
    {
        double acc = 0;
        foreach (var number in data)
        {
            acc += number;
        }

        var mean = acc / data.Length;
        return Task.FromResult(mean);
    }

    private static Task<double> CalculateStandardDeviation(double mean, double[] data)
    {
        double acc = 0;
        foreach (var number in data)
        {
            acc += (number - mean) * (number - mean);
        }
        
        var standardDeviation = Math.Sqrt(acc / data.Length);
        return Task.FromResult(standardDeviation);
    }
}