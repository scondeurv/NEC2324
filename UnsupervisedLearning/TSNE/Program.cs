﻿using System.Collections.Immutable;
using CommandLine;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Legends;
using OxyPlot.Series;
using Tools.Common;
using TSNE;

await Parser.Default.ParseArguments<Options>(args).WithParsedAsync(async opt =>
{
    var dataset = new Dataset();
    await dataset.Load(opt.InputFile, opt.Delimiter, opt.NoHeader);
    var matrix = dataset
        .ToJagged();
    var input = matrix
        .Select(row => row[0..^1])
        .ToArray();
    
    var classes = matrix
        .Select(row => row[^1])
        .ToImmutableArray();
    
    var tSNE = new Accord.MachineLearning.Clustering.TSNE
    {
        NumberOfInputs = input[0].Length,
        NumberOfOutputs = 2,
        Perplexity = opt.Perplexity,
        Theta = opt.Theta,
    };
    
    var result = tSNE.Transform(input);
    
    var plotModel = new PlotModel { Title = "t-SNE Projection" };

    var markerTypes = new List<MarkerType>
    {
        MarkerType.Circle,
        MarkerType.Square,
        MarkerType.Triangle,
        MarkerType.Diamond,
    };

    var colors = new List<OxyColor>
    {
        OxyColors.Green,
        OxyColors.Red,
        OxyColors.Blue,
        OxyColors.Yellow,
        OxyColors.Purple,
        OxyColors.Orange,
    };
    
    IDictionary<int, ScatterSeries> series = new Dictionary<int, ScatterSeries>();
    for (var row = 0; row < classes.Length; row++)
    {
        var @class = (int)classes[row];
        if (!series.ContainsKey(@class))
        {
            series.Add(@class, new ScatterSeries
            {
                MarkerType = markerTypes[@class % markerTypes.Count],
                MarkerFill = colors[@class % colors.Count],
                Title = $"Class {@class}",
            });
        }

        series[@class].Points.Add(new ScatterPoint(result[row][0], result[row][1]));
    }
    
    foreach (var scatterSeries in series.Values)
    {
        plotModel.Series.Add(scatterSeries);
    }
    
    plotModel.Background = OxyColors.White;
    plotModel.Legends.Add(new Legend
    {
        LegendPlacement = LegendPlacement.Outside,
        LegendPosition = LegendPosition.RightTop,
        LegendBackground = OxyColors.White,
        LegendBorder = OxyColors.Black,
        LegendBorderThickness = 2,
    });
    plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Embedding 1" });
    plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Embedding 2" });
    plotModel.IsLegendVisible = true;
    PngExporter.Export(plotModel, $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-tnse.png", 600, 400);    
});