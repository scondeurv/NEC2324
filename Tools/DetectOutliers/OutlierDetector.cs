using OxyPlot;
using OxyPlot.Core.Drawing;
using OxyPlot.Series;

namespace Tools.DetectOutliers;

public sealed class OutlierDetector
{
    public List<double> FindOutliersWithZScore(List<double> data)
    {
        var mean = data.Average();
        var stdDev = StandardDeviation(data);
        var threshold = 3.0;

        return data.Where(x => Math.Abs((x - mean) / stdDev) > threshold).ToList();
    }

    public List<double> FindOutliersWithIQR(List<double> data)
    {
        var sortedData = data.OrderBy(x => x).ToList();
        var q1 = sortedData[(int)(0.25 * sortedData.Count)];
        var q3 = sortedData[(int)(0.75 * sortedData.Count)];
        var iqr = q3 - q1;
        var lowerBound = q1 - 1.5 * iqr;
        var upperBound = q3 + 1.5 * iqr;

        return data.Where(x => x < lowerBound || x > upperBound).ToList();
    }

    public void ExportOutliers(List<double> data, string pngFilePath, string txtFilePath)
    {
        var outliers = FindOutliersWithIQR(data);
        var plotModel = new PlotModel { Title = "Outliers" };
        var outlierSeries = new ScatterSeries { MarkerType = MarkerType.Circle, MarkerFill = OxyColors.Red };
        var regularSeries = new ScatterSeries { MarkerType = MarkerType.Circle, MarkerFill = OxyColors.Blue };

        for (int i = 0; i < data.Count; i++)
        {
            var point = new ScatterPoint(i, data[i]);
            if (outliers.Contains(data[i]))
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
        PngExporter.Export(plotModel, pngFilePath, 600, 400);

        File.WriteAllLines(txtFilePath, outliers.Select((value, index) => $"Position: {index}, Value: {value}"));
    }

    private double StandardDeviation(List<double> data)
    {
        var mean = data.Average();
        var variance = data.Average(x => Math.Pow(x - mean, 2));
        return Math.Sqrt(variance);
    }
}