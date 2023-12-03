using Accord.IO;
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
        Dataset validationDataset = null;
        MultipleLinearRegression regression = null;
        double[] result = null;
        double[] outputs = null;
        double[] featureX = null;
        double[] featureY = null;
        var features = (opt.PlotFeatures?.Split(':') ?? dataset.Data.Keys.Take(2)).ToArray();
        
        if (opt.ModelFile != null)
        {
            Serializer.Load(opt.ModelFile, out regression);
            result = regression.Transform(dataset.Split().X);
            outputs = dataset.Split().Y;
            dataset.Data.TryGetValue(features[0], out featureX);
            dataset.Data.TryGetValue(features[1], out featureY);
        }
        else
        {
            var ols = new OrdinaryLeastSquares();
            (trainDataset, validationDataset) = dataset.SplitDataset(opt.TrainingPercentage);
            var (trainInputs, trainOutputs) = trainDataset.Split();
            regression = ols.Learn(trainInputs, trainOutputs);
            double[][] inputs = null;
            (inputs, outputs) = validationDataset.Split();
            result = regression.Transform(inputs);
            validationDataset.Data.TryGetValue(features[0], out featureX);
            validationDataset.Data.TryGetValue(features[1], out featureY);
            var modelFile = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-mlr.model";
        
            regression.Save(modelFile);
            Console.WriteLine($"Model saved to {modelFile}");

        }
        
        var predicted = result.Select(p => p > opt.Threshold ? 1.0 : 0.0).ToArray();

        var confusionMatrix = new ConfusionMatrix(predicted.Select(p => (int)p).ToArray(),
            outputs.Select(o => (int)o).ToArray());
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

        var roc = new ReceiverOperatingCharacteristic(outputs.ToArray(),
            predicted);
        roc.Compute(10000);

        Console.WriteLine($"AUC: {roc.Area}");
        
        if (opt.ExportPlots)
        {
            var fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-mlr-expected.png";
            PlotExporter.ExportScatterPlot($"Actual {dataset.Data.Keys.Last()}", features, featureX, featureY,
                outputs.Select(y => (int)y).ToArray(),
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