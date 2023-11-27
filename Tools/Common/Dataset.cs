using System.Globalization;
using Tools.Common.Scaling;

namespace Tools.Common;

public class Dataset
{
    public IReadOnlyDictionary<string, double[]> Data { get; private set; } = new Dictionary<string, double[]>(0);
    public IReadOnlyDictionary<string, double[]> ScaledData { get; private set; } = new Dictionary<string, double[]>(0);
    
    public async Task Load(string fileName, string delimiter, bool noHeader)
    {
        var data = File.ReadLinesAsync(fileName);
        var isHeader = !noHeader;
        var table = new Dictionary<string, List<double>>();
        await foreach (var row in data)
        {
            if (isHeader)
            {
                var features = row.Split(delimiter);
                foreach (var feature in features)
                {
                    table.Add(feature, new List<double>());
                }

                isHeader = false;
                continue;
            }

            if (table.Count == 0)
            {
                var index = 1;
                foreach (var item in row.Split(delimiter))
                {
                    table.Add($"Feature {index++}", new List<double>());
                }
            }

            var values = row
                .Split(delimiter)
                .Select(v => double.Parse(v, CultureInfo.InvariantCulture))
                .ToArray();
            var col = 0;
            foreach (var feature in table.Keys)
            {
                table[feature].Add(values[col++]);
            }
        }

        Data = table.ToDictionary(t => t.Key, t => t.Value.ToArray());
    }
    
    public async Task Scale(IReadOnlyDictionary<string, IScalingMethod> scalingMethodPerFeature)
    {
        var features = Data.Keys.ToArray();
        var featuresToScale = scalingMethodPerFeature.Keys.ToList();
        var scalingTasks = new List<Task<double[]>>();
        var featureIndexes = new List<int>();
        foreach (var (feature, col) in features.Select((f, i) => (f, i)))
        {
            if (featuresToScale.Contains(feature))
            {
                featureIndexes.Add(col);
                var featureData = new double[Data[feature].Length];
                for (var row = 0; row < Data[feature].Length; row++)
                {
                    featureData[row] = Data[feature][row];
                }

                var scalingMethod = scalingMethodPerFeature[feature];
                scalingTasks.Add(scalingMethod.Scale(featureData));
            }
        }

        var results = await Task.WhenAll(scalingTasks);
        var scaledData = new Dictionary<string, double[]>(Data.Count);

        foreach (var (feature, col) in features.Select((f, i) => (f, i)).ToArray())
        {
            if (featuresToScale.Contains(feature))
            {
                var scaledFeatureData = new double[Data[feature].Length];
                for (var row = 0; row < Data[feature].Length; row++)
                {
                    scaledFeatureData[row] = results[featureIndexes.IndexOf(col)][row];
                }

                scaledData.Add(feature, scaledFeatureData);
            }
            else
            {
                scaledData.Add(feature, Data[feature]);
            }
        }
        ScaledData = scaledData;
    }
    
    public async Task Save(string fileName, string delimiter, bool saveScaledData = false) 
    {
        await using var outputFile = File.CreateText(fileName);
        var features = Data.Keys.ToArray();
        await outputFile.WriteLineAsync(string.Join(delimiter, features));
        for (var row = 0; row < Data[features[0]].Length; row++)
        {
            var values = new List<string>();
            foreach (var feature in features)
            {
                values.Add(saveScaledData ? ScaledData[feature][row].ToString(CultureInfo.InvariantCulture) : Data[feature][row].ToString(CultureInfo.InvariantCulture));
            }
            await outputFile.WriteLineAsync(string.Join(delimiter, values));
        }
    }
}