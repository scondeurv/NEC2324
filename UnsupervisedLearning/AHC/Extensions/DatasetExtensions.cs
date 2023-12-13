using Tools.Common;

namespace AHC.Extensions;

public static class DatasetExtensions
{
    public static HashSet<DataPoint> ToDataPoints(this Dataset dataset, bool scaled = false)
    {
        var data = scaled ? dataset.ScaledData : dataset.Data;

        var dataPoints = new DataPoint[data.First().Value.Length];
        for (var row = 0; row < data.First().Value.Length; row++)
        {
            var values = new double[data.Count - 1];
            var col = 0;
            foreach (var feature in data.Keys)
            {
                if (feature != data.Keys.Last())
                {
                    values[col++] = data[feature][row];
                }
            }

            dataPoints[row] = new DataPoint(row.ToString(), values);
        }

        return dataPoints.ToHashSet();
    }
}