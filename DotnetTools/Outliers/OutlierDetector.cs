using OxyPlot;
using OxyPlot.Core.Drawing;
using OxyPlot.Legends;
using OxyPlot.Series;

namespace Tools.Outliers;

internal sealed class OutlierDetector
{
    public IReadOnlyDictionary<string, double[]> FindOutliersWithZScore(IReadOnlyDictionary<string, double[]> data,
        double threshold)
    {
        var outliers = new Dictionary<string, double[]>(data.Count);
        foreach (var feature in data.Keys)
        {
            var mean = data[feature].Average();
            var stdDev = StandardDeviation(data[feature]);

            outliers.Add(
                feature,
                data[feature].Where(x => Math.Abs((x - mean) / stdDev) > threshold)
                    .ToArray());
        }

        return outliers;
    }

    public IReadOnlyDictionary<string, double[]> FindOutliersWithIQR(IReadOnlyDictionary<string, double[]> data)
    {
        var outliers = new Dictionary<string, double[]>(data.Count);
        foreach (var feature in data.Keys)
        {
            var sortedData = data[feature].OrderBy(x => x).ToArray();
            var q1 = sortedData[(int)(0.25 * sortedData.Length)];
            var q3 = sortedData[(int)(0.75 * sortedData.Length)];
            var iqr = q3 - q1;
            var lowerBound = q1 - 1.5 * iqr;
            var upperBound = q3 + 1.5 * iqr;

            outliers.Add(feature,
                data[feature]
                    .Where(x => x < lowerBound || x > upperBound)
                    .ToArray());
        }

        return outliers;
    }

    public void ExportOutliers(IReadOnlyDictionary<string, double[]> data,
        IReadOnlyDictionary<string, double[]> outliers)
    {
        foreach (var feature in data.Keys)
        {
            var plotModel = new PlotModel { Title = $"{feature} - Outliers" };
            plotModel.Legends.Add(new Legend
            {
                LegendPlacement = LegendPlacement.Outside,
                LegendPosition = LegendPosition.RightTop,
                LegendBackground = OxyColors.White,
                LegendBorder = OxyColors.Black,
                LegendBorderThickness = 2,
            });
            var outlierSeries = new ScatterSeries { Title = "Outlier", MarkerType = MarkerType.Diamond, MarkerFill = OxyColors.Red, MarkerStroke = OxyColors.Black};
            var regularSeries = new ScatterSeries { Title = "Norm", MarkerType = MarkerType.Circle, MarkerFill = OxyColors.Blue , MarkerStroke = OxyColors.Black};

            for (var i = 0; i < data[feature].Length; i++)
            {
                var point = new ScatterPoint(i, data[feature][i]);
                if (outliers[feature].Contains(data[feature][i]))
                {
                    outlierSeries.Points.Add(point);
                }
                else
                {
                    regularSeries.Points.Add(point);
                }
            }

            plotModel.Series.Add(regularSeries);
            plotModel.Series.Add(outlierSeries);
            plotModel.Background = OxyColors.White;
            PngExporter.Export(plotModel, $"{feature}-outliers.png", 600, 400);

            // File.WriteAllLines($"{feature}-outliers.txt",
            //     outliers.Select((kvp, index) =>
            //         $"Position: {index}, Value: {string.Join(", ", kvp.Value.Select(v => v.ToString("F2")))}"));
        }
    }

    private static double StandardDeviation(IEnumerable<double> data)
    {
        var mean = data.Average();
        var variance = data.Average(x => Math.Pow(x - mean, 2));
        return Math.Sqrt(variance);
    }
}