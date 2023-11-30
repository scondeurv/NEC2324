using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Legends;
using OxyPlot.Series;

namespace Tools.Common;

public static class PlotExporter
{
    public static void ExportScatterPlot(string title, string[] features, double[] featureX, double[] featureY,
        int[] classes, string outputFile)
    {
        var plotModel = new PlotModel { Title = title };
        plotModel.Legends.Add(new Legend
        {
            LegendPlacement = LegendPlacement.Inside,
            LegendPosition = LegendPosition.RightTop,
            LegendBackground = OxyColors.White,
            LegendBorderThickness = 2,
        });
        plotModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom,
            Title = features[0],
            Maximum = featureX.Max(),
        });
        plotModel.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left,
            Title = features[1],
            Maximum = featureY.Max(),
        });

        var series = new Dictionary<int, ScatterSeries>();
        foreach (var @class in classes)
        {
            if (!series.ContainsKey(@class))
            {
                var (color, marker) = GetColorAndMarkerType(@class);
                var ss = new ScatterSeries
                {
                    Title = $"{@class}",
                    MarkerType = marker,
                    MarkerFill = color,
                    MarkerStroke = OxyColors.Black,
                    MarkerStrokeThickness = 2,
                };
                series.Add(@class, ss);
            }
        }

        for (var i = 0; i < featureX.Length; i++)
        {
            var point = new ScatterPoint(featureX[i], featureY[i]);
            series[classes[i]].Points.Add(point);
        }

        foreach (var ss in series.Values)
        {
            plotModel.Series.Add(ss);
        }

        plotModel.Background = OxyColors.White;
        PngExporter.Export(plotModel, outputFile, 600, 400);
    }

    public static void ExportRoc(IEnumerable<(double FPR, double TPR)> points, string fileName)
    {
        var plotModel = new PlotModel { Title = "ROC Curve" };

        plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Minimum = 0, Maximum = 1, Title = "FPR" });
        plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = 0, Maximum = 1, Title = "TPR" });

        var rocSeries = new LineSeries
        {
            Color = OxyColors.Blue,
        };

        foreach (var point in points)
        {
            rocSeries.Points.Add(new DataPoint(point.FPR, point.TPR));
        }

        plotModel.Series.Add(rocSeries);
        plotModel.Background = OxyColors.White;

        PngExporter.Export(plotModel, fileName, 600, 400);
    }


    static (OxyColor, MarkerType) GetColorAndMarkerType(int value)
    {
        // Define a dictionary with color and marker type pairs
        Dictionary<int, (OxyColor, MarkerType)> colorMarkerDict = new Dictionary<int, (OxyColor, MarkerType)>
        {
            { 0, (OxyColors.Red, MarkerType.Circle) },
            { 1, (OxyColors.Blue, MarkerType.Square) },
            { 2, (OxyColors.Green, MarkerType.Triangle) },
            { 3, (OxyColors.Yellow, MarkerType.Diamond) },
            { 4, (OxyColors.Purple, MarkerType.Plus) },
            { 5, (OxyColors.Orange, MarkerType.Star) },
            // Add more pairs as needed
        };

        // If the value exists in the dictionary, return the corresponding color and marker type
        if (colorMarkerDict.TryGetValue(value, out var colorMarkerPair))
        {
            return colorMarkerPair;
        }

        // If the value does not exist in the dictionary, return a default color and marker type
        return (OxyColors.Black, MarkerType.Cross);
    }
}