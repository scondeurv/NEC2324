using CommandLine;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Legends;
using OxyPlot.Series;
using PCA;


await Parser.Default.ParseArguments<Options>(args).WithParsedAsync(async opt =>
{
    var input = File.ReadLines(opt.InputFile).First();
    var firstLine = input.Split(opt.Delimiter);
    var columns = new List<TextLoader.Column>(firstLine.Length);
    if (opt.NoHeader)
    {
        for (var i = 0; i < firstLine.Length - 1; i++)
        {
            columns.Add(new TextLoader.Column($"X{i}", DataKind.Single, i));
        }

        columns.Add(new TextLoader.Column($"Class", DataKind.Single, firstLine.Length - 1));
    }
    else
    {
        for (var i = 0; i < firstLine.Length - 1; i++)
        {
            columns.Add(new TextLoader.Column($"{firstLine[i]}", DataKind.Single, i));
        }

        columns.Add(new TextLoader.Column($"{firstLine[^1]}", DataKind.Single, firstLine.Length - 1));
    }

    var context = new MLContext();
    var data = context.Data.LoadFromTextFile(opt.InputFile, hasHeader: !opt.NoHeader, separatorChar: opt.Delimiter[0],
        columns: columns.ToArray());
    var pipeline = context.Transforms
        .Concatenate("Features", columns.Take(columns.Count - 1).Select(c => c.Name).ToArray())
        .Append(context.Transforms.ProjectToPrincipalComponents("Projection", "Features", rank: columns.Count - 1));

    var model = pipeline.Fit(data);
    
    var transformedData = model.Transform(data);
    
    var pcaData = context.Data
        .CreateEnumerable<PcaResult>(transformedData, reuseRowObject: false)
        .ToArray();
    
    var groupedData = pcaData
        .GroupBy(p => p.@class)
        .OrderBy(g => g.Key);

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

    var index = 0;
    foreach (var group in groupedData)
    {
        var scatterSeries = new ScatterSeries
        {
            MarkerType = markerTypes[index % markerTypes.Count],
            MarkerFill = colors[index % colors.Count],
            Title = $"Class {group.Key}",
        };

        var points = group.Select(p => new ScatterPoint(p.Projection[0], p.Projection[1])).ToList();
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
});