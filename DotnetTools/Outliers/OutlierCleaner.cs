namespace Tools.Outliers;

public sealed class OutlierCleaner
{
    public IReadOnlyDictionary<string, double[]> CleanOutliers(IReadOnlyDictionary<string, double[]> data,
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
}