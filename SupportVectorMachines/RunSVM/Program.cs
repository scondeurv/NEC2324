using Accord.Statistics.Analysis;
using CommandLine;
using LibSVMsharp;
using LibSVMsharp.Extensions;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Legends;
using OxyPlot.Series;
using SupportVectorMachines.RunSVM;
using SupportVectorMachines.RunSVM.Extensions;
using Tools.Common;

await Parser.Default.ParseArguments<Options>(args)
    .WithParsedAsync(async opt =>
    {
        var dataset = new Dataset();
        await dataset.Load(opt.DatasetFile, opt.Delimiter, opt.NoHeader);
        
        Dataset trainDataset = null;
        Dataset testDataset = null;
        if (opt.TestFile != null)
        {
            testDataset = new Dataset();

            await testDataset.Load(opt.TestFile, opt.Delimiter, opt.NoHeader);
        }
        else
        {
            (trainDataset, testDataset) = dataset.SplitDataset(opt.TrainingPercentage);
        }

        ConfusionMatrix confusionMatrix = null;
        SVMModel model = null;
        SVMParameter parameters = null;
        int[] predicted = null;
        int[] expected = null;
        if (opt.ModelFile != null)
        {
            model = SVM.LoadModel(opt.ModelFile);
            (confusionMatrix, expected, predicted) = SVMRunner.Predict(model, dataset.ToSVMProblem());
            if (opt.ExportPlots)
            {
                var features = (opt.PlotFeatures?.Split(':') ?? dataset.Data.Keys.Take(2)).ToArray();
                dataset.Data.TryGetValue(features[0], out var featureX);
                dataset.Data.TryGetValue(features[1], out var featureY);
                var fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-expected.png";
                ExportScatterPlot($"Actual {dataset.Data.Keys.Last()}", features, featureX, featureY, expected, fileName);
                Console.WriteLine("Exported plot to {0}", fileName);
                fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-predicted.png";
                ExportScatterPlot($"Predicted {dataset.Data.Keys.Last()}", features, featureX, featureY, predicted, fileName);
                Console.WriteLine("Exported plot to {0}", fileName);
            }
        }
        else
        {
            var runner = new SVMRunner();
            var trainProblem = trainDataset.ToSVMProblem();
            var testProblem = testDataset.ToSVMProblem();

            var optimizer = opt.Optimizer switch
            {
                "random" => SVMRunner.SVMOptimizer.RandomSearch,
                "search" => SVMRunner.SVMOptimizer.GridSearch,
                _ => throw new ArgumentException($"Invalid optimizer: {opt.Optimizer}")
            };
            var svmType = opt.Svc switch
            {
                "c" => SVMType.C_SVC,
                "nu" => SVMType.NU_SVC,
                _ => throw new ArgumentException($"Invalid SVC type: {opt.Svc}")
            };
            var kernelType = opt.Kernel switch
            {
                "linear" => SVMKernelType.LINEAR,
                "poly" => SVMKernelType.POLY,
                "rbf" => SVMKernelType.RBF,
                "sigmoid" => SVMKernelType.SIGMOID,
                _ => throw new ArgumentException($"Invalid kernel type: {opt.Kernel}")
            };
            (parameters, model, confusionMatrix) = runner.RunSVC(svmType, trainProblem, testProblem,
                optimizer: optimizer, iterations: opt.Iterations, fScoreTarget: opt.FScore, kernelType: kernelType);

            var modelFile = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-model.txt";
            model.SaveModel(modelFile);
            Console.WriteLine($"Model saved to {modelFile}");
        }
        
        Console.WriteLine("Confusion Matrix:");
        Console.WriteLine("-----------------");
        Console.WriteLine(
            $"| TP: {confusionMatrix.TruePositives} | FP: {confusionMatrix.FalsePositives} |");
        Console.WriteLine("-----------------");
        Console.WriteLine(
            $"| FN: {confusionMatrix.FalseNegatives} | TN: {confusionMatrix.TrueNegatives} |");
        Console.WriteLine("-----------------\n");
        Console.WriteLine($"Accuracy: {confusionMatrix.Accuracy}");
        Console.WriteLine($"Precision: {confusionMatrix.Precision}");
        Console.WriteLine($"Sensitivity: {confusionMatrix.Sensitivity}");
        Console.WriteLine($"FScore: {confusionMatrix.FScore}");
        if (model.Parameter.Type == SVMType.C_SVC)
        {
            Console.WriteLine($"C: {model.Parameter.C}");
        }

        if (model.Parameter.Type == SVMType.NU_SVC)
        {
            Console.WriteLine($"Nu: {model.Parameter.Nu}");
        }

        Console.WriteLine($"Gamma: {model.Parameter.Gamma}");
        Console.WriteLine($"Degree: {model.Parameter.Degree}");
    });


static void ExportScatterPlot(string title, string[] features, double[] featureX, double[] featureY, int[] classes, string outputFile)
{
    var plotModel = new PlotModel {Title = title};
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