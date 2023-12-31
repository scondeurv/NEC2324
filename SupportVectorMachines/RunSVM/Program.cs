﻿using Accord.Statistics.Analysis;
using CommandLine;
using LibSVMsharp;
using LibSVMsharp.Extensions;
using SupportVectorMachines.RunSVM;
using SupportVectorMachines.RunSVM.Extensions;
using Tools.Common;

await Parser.Default.ParseArguments<Options>(args)
    .WithParsedAsync(async opt =>
    {
        var dataset = new Dataset();
        await dataset.Load(opt.DatasetFile, opt.Delimiter, opt.NoHeader);
        
        ConfusionMatrix confusionMatrix = null;
        SVMModel model = null;
        SVMParameter parameters = null;
        int[] predicted = null;
        int[] expected = null;
        double[] featureX = null;
        double[] featureY = null;
        Dataset validationDataset = null;
        Dataset trainDataset = null;
        if (opt.ModelFile != null)
        {
            model = SVM.LoadModel(opt.ModelFile);
            (confusionMatrix, expected, predicted) = SVMRunner.Predict(model, dataset.ToSVMProblem());
        }
        else
        {
            (trainDataset, validationDataset) = dataset.SplitDataset(opt.TrainingPercentage);
            var runner = new SVMRunner();
            var trainProblem = trainDataset.ToSVMProblem();
            var validationProblem = validationDataset.ToSVMProblem();

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
            (parameters, model, confusionMatrix, expected, predicted) = runner.RunSVC(svmType, trainProblem, validationProblem,
                optimizer: optimizer, iterations: opt.Iterations, fScoreTarget: opt.FScore, kernelType: kernelType);

            var modelFile = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-svm.model";
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
        
        var classificationError = 100.0*(confusionMatrix.FalseNegatives + confusionMatrix.FalsePositives)/
                                  (confusionMatrix.TruePositives + confusionMatrix.TrueNegatives +
                                   confusionMatrix.FalseNegatives + confusionMatrix.FalsePositives);
        Console.WriteLine($"Classification Error (%): {classificationError}");
        
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
        var roc = new ReceiverOperatingCharacteristic(expected.Select(p => p > 0).ToArray(),
            predicted);
        roc.Compute(1000);
        Console.WriteLine($"AUC: {roc.Area}");        
        
        if (opt.ExportPlots)
        {
            var features = (opt.PlotFeatures?.Split(':') ?? dataset.Data.Keys.Take(2)).ToArray();

            if (opt.ModelFile != null)
            {
                dataset.Data.TryGetValue(features[0], out featureX);
                dataset.Data.TryGetValue(features[1], out featureY);
            }
            else
            {
                validationDataset.Data.TryGetValue(features[0], out featureX);
                validationDataset.Data.TryGetValue(features[1], out  featureY);
            }

            var fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-svm-expected.png";
            PlotExporter.ExportScatterPlot($"Actual {dataset.Data.Keys.Last()}", features, featureX, featureY,
                expected,
                fileName);
            Console.WriteLine($"Exported plot to {fileName}");
                
            fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-svm-predicted.png";
            PlotExporter.ExportScatterPlot($"Predicted {dataset.Data.Keys.Last()}", features, featureX, featureY,
                predicted,
                fileName);
            Console.WriteLine($"Exported plot to {fileName}");
            
            fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-svm-roc.png";
            var points = roc.Points.Select(p =>
                (p.FalsePositiveRate, (double)p.TruePositives / (p.TruePositives + p.FalseNegatives)));
            PlotExporter.ExportRoc(points, $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-roc.png");
            Console.WriteLine($"Exported ROC plot to {fileName}");
        }
    });