namespace BackPropagation.Scaling;

public sealed class MinMax : IScalingMethod
{
    private (double Min, double Max) Range { get; init; }

    public MinMax((double Min, double Max) range)
    {
        Range = range.Min < range.Max 
            ? range
            : throw new ArgumentException(nameof(range));
    }
    
    public async Task<double[]> Scale(double[] data)
    {
        if (data is null || data.Length == 0)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var minMax = await Task.WhenAll(GetMin(data), GetMax(data));
        var min = minMax[0];
        var max = minMax[1];
        var scaledData = new double[data.Length];
        var deltaRange = Range.Max - Range.Min;
        var deltaMinMax = max - min;
        for (var i = 0; i < data.Length; i++)
        {
            scaledData[i] = Range.Min + deltaRange/deltaMinMax * (data[i] - min);
        }

        return scaledData;
    }

    private static Task<double> GetMin(double[] data)
    {
        double? min = null;
        foreach (var number in data)
        {
            min = min is null || number < min ? number : min;
        }

        return Task.FromResult(min.Value);
    }

    private static Task<double> GetMax(double[] data)
    {
        double? max = null;
        foreach (var number in data)
        {
            max = max is null || number > max ? number : max;
        }

        return Task.FromResult(max.Value);
    }
}