using Accord.Statistics.Analysis;
using Accord.Statistics.Models.Regression.Linear;
using CommandLine;
using SupportVectorMachines.MLR;
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
            trainDataset = dataset;
            testDataset = new Dataset();
            await testDataset.Load(opt.TestFile, opt.Delimiter, opt.NoHeader);
        }
        else
        {
            (trainDataset, testDataset) = dataset.SplitDataset(opt.TrainingPercentage);
        }

        var ols = new OrdinaryLeastSquares();

        var (trainInputs, trainOutputs) = trainDataset.Split();
        var regression = ols.Learn(trainInputs, trainOutputs);
        var (testInputs, testOutputs) = testDataset.Split();
        var result = regression.Transform(testInputs);
        var predicted = result.Select(p => p > opt.Threshold ? 1.0 : 0.0).ToArray();

        var confusionMatrix = new ConfusionMatrix(predicted.Select(p => (int)p).ToArray(),
            testOutputs.Select(o => (int)o).ToArray());
        Console.WriteLine("Confusion Matrix:");
        Console.WriteLine("-----------------");
        Console.WriteLine(
            $"| TP: {confusionMatrix.TruePositives} | FP: {confusionMatrix.FalsePositives} |");
        Console.WriteLine("-----------------");
        Console.WriteLine(
            $"| FN: {confusionMatrix.FalseNegatives} | TN: {confusionMatrix.TrueNegatives} |");
        Console.WriteLine("-----------------");
        Console.WriteLine($"Accuracy: {confusionMatrix.Accuracy}");
        Console.WriteLine($"Precision: {confusionMatrix.Precision}");
        Console.WriteLine($"Sensitivity: {confusionMatrix.Sensitivity}");
        Console.WriteLine($"FScore: {confusionMatrix.FScore}");
        
        var classificationError = 100.0*(confusionMatrix.FalseNegatives + confusionMatrix.FalsePositives)/
                                  (confusionMatrix.TruePositives + confusionMatrix.TrueNegatives +
                                            confusionMatrix.FalseNegatives + confusionMatrix.FalsePositives);
        Console.WriteLine($"Classification Error (%): {classificationError}");

        var roc = new ReceiverOperatingCharacteristic(testOutputs.ToArray(),
            predicted);
        roc.Compute(10000);

        Console.WriteLine($"AUC: {roc.Area}");
        if (opt.ExportPlots)
        {
            var features = (opt.PlotFeatures?.Split(':') ?? dataset.Data.Keys.Take(2)).ToArray();
            testDataset.Data.TryGetValue(features[0], out var featureX);
            testDataset.Data.TryGetValue(features[1], out var featureY);

            var fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-mlr-expected.png";
            PlotExporter.ExportScatterPlot($"Actual {dataset.Data.Keys.Last()}", features, featureX, featureY,
                testOutputs.Select(y => (int)y).ToArray(),
                fileName);
            Console.WriteLine($"Exported plot to {fileName}");

            fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-mlr-predicted.png";
            PlotExporter.ExportScatterPlot($"Predicted {dataset.Data.Keys.Last()}", features, featureX, featureY,
                predicted.Select(p => (int)p).ToArray(),
                fileName);
            Console.WriteLine($"Exported plot to {fileName}");

            fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-mlr-roc.png";
            var points = roc.Points.Select(p =>
                (p.FalsePositiveRate, (double)p.TruePositives / (p.TruePositives + p.FalseNegatives)));
            PlotExporter.ExportRoc(points, fileName);
            Console.WriteLine($"Exported ROC plot to {fileName}");
        }
    });