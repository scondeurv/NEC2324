using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Math.Optimization.Losses;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Statistics.Analysis;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression.Linear;
using CommandLine;
using SupportVectorMachines.BP;
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
        
        var cv = CrossValidation.Create(

            k: 10, // We will be using k = 10 folds

            learner: (p) => new SequentialMinimalOptimization<Gaussian>()
            {
                Complexity = 100 // Complexity parameter C
            },

            // How to split the data into training/validation
            splitter: (indices) => new KFoldSplitter<Gaussian>(k: 10).Split(indices),

            // Define the fitting function
            fit: (teacher, x, y, w) => teacher.Run(new SupportVectorMachine<Gaussian>(inputs: 2), x, y),

            // Define the testing function
            loss: (actual, expected, m) => new ZeroOneLoss(expected).Loss(actual),

            // Define how to compute the expected outputs for the machine
            compute: (svm, x) => svm.Decide(x)
        );

// Compute the cross-validation
        var result = cv.Compute(trainDataset.ToSVMProblem());

// Get the cross-validation performance
        double crossValidationPerformance = result.Mean;
        
        var numberOfInputs = dataset.Data.Count - 1;
        var layers = opt.Layers.Split(':').Select(int.Parse).ToArray();
        var activationInfo = GetActivationFunction(opt.ActivationFunction);
        
        var network = new ActivationNetwork(activationInfo, numberOfInputs, layers);
        var bp = new BackPropagationLearning(network)
        {
            LearningRate = opt.LearningRate, 
            Momentum = opt.Momentum, 
        };

        
// Train the network using the teacher
        var error = 1.0;
        while (error > targetError)
        {
            error = bp.Run();
        }

        var confusionMatrix = new ConfusionMatrix(predicted.Select(p => (int)p).ToArray(),
            testOutputs.Select(o => (int)o).ToArray());
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
            PlotExporter.ExportRoc(points, $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-roc.png");
            Console.WriteLine($"Exported ROC plot to {fileName}");
        }
    });
    
    static IActivationFunction GetActivationFunction(string activationFunction)
    {
        return activationFunction switch
        {
            "sigmoid" => new SigmoidFunction(),
            "tanh" => new BipolarSigmoidFunction(),
            "relu" => new RectifiedLinearFunction(),
            "linear" => new LinearFunction(),
            _ => throw new ArgumentOutOfRangeException(nameof(activationFunction), activationFunction,
                "Invalid activation function.")
        };
    }