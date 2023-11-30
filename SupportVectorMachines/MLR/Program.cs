using Accord.Statistics.Analysis;
using Accord.Statistics.Models.Regression.Linear;
using CommandLine;
using MLR;
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
        var predicted = result.Select(p => p > 0.489 ? 1.0 : 0.0).ToArray();
        
        var confusionMatrix = new ConfusionMatrix(predicted.Select(p => (int)p).ToArray(), testOutputs.Select(o => (int)o).ToArray());
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
        
        var roc = new ReceiverOperatingCharacteristic(testOutputs.ToArray(),
            predicted);
        roc.Compute(1000);
        // var optimalThreshold = FindOptimalThreshold(roc);
        // predicted = result.Select(p => p > optimalThreshold ? 1.0 : 0.0).ToArray();
        // roc = new ReceiverOperatingCharacteristic(testOutputs.ToArray(),
        //     predicted);
        // roc.Compute(1000);
        
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
            PlotExporter.ExportRoc(points, $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-roc.png");
            Console.WriteLine($"Exported ROC plot to {fileName}");
        }
    });
    
static double FindOptimalThreshold(ReceiverOperatingCharacteristic roc)
{
    var maxJ = double.NegativeInfinity;
    var optimalThreshold = 0.0;

    // Iterate over the possible thresholds
    for (var threshold = 0.0; threshold <= 1; threshold += 0.01)
    {
        // Calculate the sensitivity and specificity for the current threshold
        var sensitivity = roc.Points.Where(p => p.Sensitivity >= threshold).Max(p => p.Sensitivity);
        var specificity = roc.Points.Where(p => p.FalsePositiveRate <= (1 - threshold)).Max(p => 1 - p.FalsePositiveRate);

        // Calculate Youden's J statistic
        var j = sensitivity + specificity - 1;

        // Update the optimal threshold if the current J statistic is greater than the maximum J statistic found so far
        if (j > maxJ)
        {
            maxJ = j;
            optimalThreshold = threshold;
        }
    }

    return optimalThreshold;
}