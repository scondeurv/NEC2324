using System.Collections.Immutable;
using CommandLine;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Legends;
using OxyPlot.Series;
using PCA;

Parser.Default.ParseArguments<Options>(args).WithParsed(opt =>
{
    var pca = new PrincipalComponentAnalyzer();
    var (pcaResults, variances) = pca.Run(opt.InputFile, opt.Delimiter, opt.NoHeader);

    var plotModel = new PlotModel { Title = "PCA 2D Projection" };

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

    var groupedResults = pcaResults.GroupBy(r => r.@class).ToDictionary(g => g.Key, g => g.ToList());
    var index = 0;
    foreach (var group in groupedResults)
    {
        var scatterSeries = new ScatterSeries
        {
            MarkerType = markerTypes[index % markerTypes.Count],
            MarkerFill = colors[index % colors.Count],
            Title = $"Class {group.Key}",
        };

        var points = group.Value.Select(p => new ScatterPoint(p.Projection[0], p.Projection[1])).ToList();
        scatterSeries.Points.AddRange(points);

        plotModel.Series.Add(scatterSeries);
        index++;
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
    plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "PC1" });
    plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "PC2" });
    plotModel.IsLegendVisible = true;
    PngExporter.Export(plotModel, $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-pca-2d.png", 600, 400);

    plotModel = new PlotModel { Title = "Accumulated Variance" };
    plotModel.Background = OxyColors.White;

    var lineSeries = new LineSeries();
    var accumulatedVariance = 0d;
    var totalVariance = variances.Sum();
    for (var i = 0; i < variances.Count; i++)
    {
        lineSeries.Points.Add(new DataPoint(i + 1, 100 * (accumulatedVariance += variances[i] / totalVariance)));
    }

    lineSeries.Color = OxyColors.Red;
    plotModel.Series.Add(lineSeries);
    plotModel.Axes.Add(new LinearAxis
        { Position = AxisPosition.Bottom, Title = "Principal Component", MinorStep = 1, MajorStep = 1 });
    plotModel.Axes.Add(new LinearAxis
        { Position = AxisPosition.Left, Title = "Accumulated Variance (%)", MinorStep = 10, MajorStep = 10 });

    PngExporter.Export(plotModel, $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-pca-av.png", 600, 400);
});