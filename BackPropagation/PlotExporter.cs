﻿using OxyPlot;
using OxyPlot.Annotations;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Series;

namespace BackPropagation;

public class PlotExporter
{
    public void ExportLinear(string title, string x, string y, IReadOnlyDictionary<string, (double X, double Y)[]> data,
        string outputFile, string? annotation = null)
    {
        var series = new List<Series>();
        foreach (var item in data)
        {
            var lineSeries = new LineSeries
            {
                Title = item.Key,
                LegendKey = item.Key,
            };

            foreach (var point in item.Value)
            {
                lineSeries.Points.Add(new DataPoint(point.X, point.Y));
            }

            series.Add(lineSeries);
        }

        ExportPlot(series, title, x, y, outputFile, annotation);
    }

    public void ExportScatter(string title, string x, string y, (double X, double Y)[] data, string outputFile,
        string? annotation = null)
    {
        var series = new ScatterSeries()
        {
            MarkerType = MarkerType.Circle,
        };

        foreach (var point in data)
        {
            series.Points.Add(new ScatterPoint(point.X, point.Y));
        }

        ExportPlot(new[] { series }, title, x, y, outputFile, annotation);
    }

    private void ExportPlot(IEnumerable<Series> series, string title, string x, string y, string outputFile,
        string? annotation = null)
    {
        var plotModel = new PlotModel { Title = title };
        plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = x });
        plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = y });
        plotModel.Background = OxyColor.FromRgb(255, 255, 255);

        foreach (var s in series)
        {
            plotModel.Series.Add(s);
        }

        if (annotation != null)
        {
            plotModel.Annotations.Add(new TextAnnotation
            {
                Text = annotation,
                TextPosition = new DataPoint(500, 100),
            });
        }

        var pngExporter = new PngExporter { Width = 1024, Height = 768 };
        Directory.CreateDirectory($".{Path.DirectorySeparatorChar}/output");
        pngExporter.ExportToFile(plotModel, $"output{Path.DirectorySeparatorChar}{outputFile}.png");
    }
}