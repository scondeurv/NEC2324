namespace Tools.Common.Scaling;

public sealed class MinMaxScaler : IScaler
{
    private (double Min, double Max)? OriginalRange { get; set; }
    private (double Min, double Max) ScaledRange { get; }

    public MinMaxScaler(double rangeMin, double rangeMax)
    {
        ScaledRange = rangeMin < rangeMax
            ? (rangeMin, rangeMax)
            : throw new ArgumentException(nameof(rangeMax));
    }
    
    public async Task<double[]> Scale(double[] data, CancellationToken? cancellationToken = null)
    {
        if (data is null || data.Length == 0)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var minMax = await Task.WhenAll(GetMin(data, cancellationToken), GetMax(data, cancellationToken));
        OriginalRange = (minMax[0], minMax[1]);
        var scaledData = new double[data.Length];
        var deltaScaledRange = ScaledRange.Max - ScaledRange.Min;
        var deltaMinMax = OriginalRange.Value.Max - OriginalRange.Value.Min;
        for (var i = 0; i < data.Length; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            scaledData[i] = ScaledRange.Min + deltaScaledRange/deltaMinMax * (data[i] - OriginalRange.Value.Min);
        }

        return scaledData;
    }
    
    public Task<double[]> Descale(double[] scaledData, CancellationToken? cancellationToken = null)
    {
        if (OriginalRange is null)
        {
            throw new NotSupportedException("Scale must be run first!");
        }
        
        if (scaledData is null || scaledData.Length == 0)
        {
            throw new ArgumentNullException(nameof(scaledData));
        }
        
        var descaledData = new double[scaledData.Length];
        var deltaScaledRange = ScaledRange.Max - ScaledRange.Min;
        var deltaMinMax = OriginalRange.Value.Max - OriginalRange.Value.Min;
        for (var i = 0; i < scaledData.Length; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            descaledData[i] = OriginalRange.Value.Min + deltaMinMax/deltaScaledRange * (scaledData[i] - ScaledRange.Min);
        }

        return Task.FromResult(descaledData);
    }

    private static Task<double> GetMin(double[] data, CancellationToken? cancellationToken)
    {
        double? min = null;
        foreach (var number in data)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            min = min is null || number < min ? number : min;
        }

        return Task.FromResult(min.Value);
    }

    private static Task<double> GetMax(double[] data, CancellationToken? cancellationToken)
    {
        double? max = null;
        foreach (var number in data)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            max = max is null || number > max ? number : max;
        }

        return Task.FromResult(max.Value);
    }
}