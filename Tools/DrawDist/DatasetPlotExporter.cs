﻿using System.Globalization;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Statistics;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Series;

namespace DrawDist;

public sealed class DatasetPlotExporter
{
    public async Task Export(Options opt)
    {
        var dataset = await ReadFile(opt.InputFile, opt.Delimiter, opt.NoHeader);
        await Task
            .WhenAll(
                CreateHistogramSeries(dataset, out var histogramSeries),
                CreateNormalSeries(dataset, 0.01, out var normalSeries)
            );

        string? fileName;
        foreach (var item in histogramSeries)
        {
            var plotModel = new PlotModel
            {
                Title = item.Key,
                Background = OxyColors.White,
                Axes =
                {
                    new LinearAxis
                    {
                        Position = AxisPosition.Bottom,
                        Minimum = dataset[item.Key].Min(),
                        Maximum = dataset[item.Key].Max(),
                        Title = item.Key,
                    },
                    new LinearAxis
                    {
                        Position = AxisPosition.Left,
                        Minimum = 0,
                        Maximum = item.Value.Items.Max(i => i.Value) * 1.2,
                        Title = "Count",
                        PositionTier = 0,
                        Key = "HistogramAxis",
                        IsAxisVisible = false
                    },
                    new LinearAxis
                    {
                        Position = AxisPosition.Left,
                        Minimum = 0,
                        Maximum = normalSeries[item.Key].Points.Max(p => p.Y) * 1.2,
                        Title = "Density",
                        PositionTier = 1,
                        Key = "DensityAxis",
                    },
                }
            };
            item.Value.YAxisKey = "HistogramAxis";
            normalSeries[item.Key].YAxisKey = "DensityAxis";
            plotModel.Series.Add(item.Value);
            plotModel.Series.Add(normalSeries[item.Key]);
            fileName = $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-{item.Key}";
            PngExporter.Export(plotModel, $"{fileName}.png", 600,
                400);
        }
        
        fileName = $"{Path.GetFileNameWithoutExtension(opt.InputFile)}";
        await ExportStatistics(dataset, fileName);
    }

    public async Task ExportStatistics(IReadOnlyDictionary<string, double[]> dataset, string outputFile)
    {
        using var writer = new StreamWriter($"{outputFile}-stats.csv");
        await writer.WriteLineAsync(",Min,Max,Mean,StdDev");
        foreach (var feature in dataset.Keys)
        {
            await writer.WriteLineAsync(
                $"{feature},{ArrayStatistics.Minimum(dataset[feature]).ToString(CultureInfo.InvariantCulture)},{ArrayStatistics.Maximum(dataset[feature]).ToString(CultureInfo.InvariantCulture)},{ArrayStatistics.Mean(dataset[feature]).ToString(CultureInfo.InvariantCulture)},{ArrayStatistics.StandardDeviation(dataset[feature]).ToString(CultureInfo.InvariantCulture)}");
        }
        await writer.FlushAsync();
    }

    private static Task CreateNormalSeries(IReadOnlyDictionary<string, double[]> dataset,
        double resolution, out Dictionary<string, LineSeries> normalSeries)
    {
        normalSeries = new Dictionary<string, LineSeries>(dataset.Keys.Count());
        foreach (var feature in dataset.Keys)
        {
            var mean = ArrayStatistics.Mean(dataset[feature]);
            var stdDev = ArrayStatistics.StandardDeviation(dataset[feature]);
            var normalDist = new Normal(mean, stdDev);
            var min = dataset[feature].Min();
            var max = dataset[feature].Max();

            var series = new LineSeries
            {
                Title = "Normal Distribution",
                Color = OxyColor.FromRgb(255,140,0)
            };

            for (var x = min; x <= max; x += resolution)
            {
                series.Points.Add(new DataPoint(x, normalDist.Density(x)));
            }

            normalSeries.Add(feature, series);
        }

        return Task.CompletedTask;
    }

    private static Task CreateHistogramSeries(IReadOnlyDictionary<string, double[]> dataset,
        out Dictionary<string, HistogramSeries> histogramSeries)
    {
        histogramSeries = new Dictionary<string, HistogramSeries>(dataset.Keys.Count());
        foreach (var feature in dataset.Keys)
        {
            var series = new HistogramSeries();
            var values = dataset[feature];
            var min = values.Min();
            var (numberOfBins, binWidth) = CalculateBins(dataset[feature]);

            for (var i = 0; i < numberOfBins; i++)
            {
                var binStart = min + (i * binWidth);
                var binEnd = min + ((i + 1) * binWidth);
                var count = values.Count(v => v >= binStart && v < binEnd);
                var bin = new HistogramItem(binStart, binEnd, binWidth * count, 1);
                series.Items.Add(bin);
            }

            series.FillColor = OxyColor.FromRgb(173,216,230);
            histogramSeries.Add(feature, series);
        }

        return Task.FromResult<IReadOnlyDictionary<string, HistogramSeries>>(histogramSeries);
    }

    private static (int NumberOfBins, double binWidth) CalculateBins(IReadOnlyCollection<double> data,
        double binWidthFactor = 0.15)
    {
        if (data.All(x => x is 0 or 1))
        {
            return (2, 0.5 * binWidthFactor);
        }
        else
        {
            var iqr = data.InterquartileRange();
            if (iqr > 0)
            {
                var binWidth = 2 * (iqr / Math.Pow(data.Count, 1 / 3.0)) * binWidthFactor;
                var range = data.Maximum() - data.Minimum();
                var binCount = (int)Math.Ceiling(range / binWidth);
                return (binCount, binWidth);
            }
            else
            {
                var binCount = (int)Math.Ceiling(Math.Sqrt(data.Count));
                var range = data.Maximum() - data.Minimum();
                var binWidth = range / binCount * binWidthFactor;
                return (binCount, binWidth);
            }
        }
    }

    private static async Task<IReadOnlyDictionary<string, double[]>> ReadFile(string fileName, string delimiter,
        bool noHeader)
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

        return table.ToDictionary(t => t.Key, t => t.Value.ToArray());
    }
}