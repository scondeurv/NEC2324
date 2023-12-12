using CommandLine;
using KMeans;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Legends;
using OxyPlot.Series;
using PCA;
using Options = KMeans.Options;

await Parser.Default.ParseArguments<Options>(args).WithParsedAsync(async opt =>
{
    var classifier = new KMeansClassifier();
    var predictedClasses = await classifier.Classify(opt.InputFile, opt.Separator, opt.NoHeader, opt.K, opt.Tolerance, opt.DistanceMethod);
    var pca = new PrincipalComponentAnalyzer();
    var (pcaResults, _) = await pca.Run(opt.InputFile, opt.Separator, opt.NoHeader);

    var plotModel = new PlotModel { Title = "k-means" };

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

    var confusionMatrix = new Dictionary<int, (int Success, int Failed)>();
    IDictionary<int, ScatterSeries> series = new Dictionary<int, ScatterSeries>();
    for (var row = 0; row < pcaResults.Length; row++)
    {
        var @class = predictedClasses[row];
        if (!series.ContainsKey(@class))
        {
            series.Add(@class, new ScatterSeries
            {
                MarkerType = markerTypes[@class % markerTypes.Count],
                MarkerFill = colors[@class % colors.Count],
                Title = $"Class {@class}",
            });

            confusionMatrix.Add(@class, (0, 0));
        }

        confusionMatrix[@class] = (
            ((int)pcaResults[row].@class) == @class ? confusionMatrix[@class].Success + 1 : confusionMatrix[@class].Success,
            ((int)pcaResults[row].@class) != @class ? confusionMatrix[@class].Failed + 1 : confusionMatrix[@class].Failed);
        series[@class].Points.Add(new ScatterPoint(pcaResults[row].Projection[0], pcaResults[row].Projection[1]));
    }

    for (var i = 1; i <= series.Count; i++)
    {
        plotModel.Series.Add(series[i]);
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
    plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "PCA Component 1" });
    plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "PCA Component 2" });
    plotModel.IsLegendVisible = true;
    PngExporter.Export(plotModel, $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-kmeans-{opt.K}.png", 600, 400);

    Console.WriteLine($"Confusion Matrix for k-means with k={opt.K}");
    Console.WriteLine($"Class\tSuccess\tFailed");
    foreach (var (key, value) in confusionMatrix.OrderBy(m => m.Key))
    {
        Console.WriteLine($"{key}\t{value.Success}\t{value.Failed}");
    }
});