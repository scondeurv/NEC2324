using BackPropagation.Exceptions;
using BackPropagation.Scaling;

namespace BackPropagation;

public class DataScaler
{
    public async Task<double[][]> Scale(double[][] data, string[] features,
        IReadOnlyDictionary<string, IScalingMethod> scalingMethodPerFeature,
        CancellationToken? cancellationToken = null)
    {
        var featuresToScale = scalingMethodPerFeature.Keys.ToList();
        var scalingTasks = new List<Task<double[]>>();
        var featureIndexes = new List<int>();
        foreach (var (feature, col) in features.Select((f, i) => (f, i)))
        {
            cancellationToken?.ThrowIfCancellationRequested();

            if (featuresToScale.Contains(feature))
            {
                featureIndexes.Add(col);
                var featureData = new double[data.Length];
                for (var row = 0; row < data.Length; row++)
                {
                    featureData[row] = data[row][col];
                }

                var scalingMethod = scalingMethodPerFeature[feature];
                scalingTasks.Add(scalingMethod.Scale(featureData, cancellationToken));
            }
        }

        var results = await Task.WhenAll(scalingTasks);
        var scaledData = new double[data.Length][];
        for (var row = 0; row < data.Length; row++)
        {
            scaledData[row] = new double[data[0].Length];
            
            for (var col = 0; col < data[0].Length; col++)
            {
                if (featureIndexes.Contains(col))
                {
                    scaledData[row][col] = results[featureIndexes.IndexOf(col)][row];
                }
                else
                {
                    scaledData[row][col] = data[row][col];
                }
            }
        }

        return scaledData;
    }
}