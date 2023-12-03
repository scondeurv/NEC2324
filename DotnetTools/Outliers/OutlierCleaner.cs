namespace Tools.Outliers;

public sealed class OutlierCleaner
{
    public IReadOnlyDictionary<string, double[]> DropOutliers(IReadOnlyDictionary<string, double[]> data,
        IReadOnlyDictionary<string, double[]> outliers)
    {
        var cleanedData = new Dictionary<string, List<double>>(data.Count);
        foreach (var feature in data.Keys)
        {
            cleanedData.Add(feature, new List<double>());
        }

        var length = data.First().Value.Length;
        for (var i = 0; i < length; i++)
        {
            var isOutlier = false;
            foreach (var feature in data.Keys)
            {
                if (outliers[feature].Contains(data[feature][i]))
                {
                    isOutlier = true;
                    break;
                }
            }

            if (!isOutlier)
            {
                foreach (var feature in data.Keys)
                {
                    cleanedData[feature].Add(data[feature][i]);
                }
            }
        }

        return cleanedData.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.ToArray());
    }
    
    public IReadOnlyDictionary<string, double[]> WinsorizeOutliers(IReadOnlyDictionary<string, double[]> data,
        IReadOnlyDictionary<string, double[]> outliers, double lowerPercentile, double upperPercentile)
    {
        var winsorizedData = new Dictionary<string, List<double>>(data.Count);
        foreach (var feature in data.Keys)
        {
            var sortedData = data[feature].OrderBy(x => x).ToArray();
            var lowerBound = sortedData[(int)(lowerPercentile * sortedData.Length)];
            var upperBound = sortedData[(int)(upperPercentile * sortedData.Length)];

            var featureData = new List<double>();
            foreach (var value in data[feature])
            {
                if (outliers[feature].Contains(value))
                {
                    if (value < lowerBound)
                    {
                        featureData.Add(lowerBound);
                    }
                    else if (value > upperBound)
                    {
                        featureData.Add(upperBound);
                    }
                }
                else
                {
                    featureData.Add(value);
                }
            }

            winsorizedData.Add(feature, featureData);
        }

        return winsorizedData.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.ToArray());
    }
}