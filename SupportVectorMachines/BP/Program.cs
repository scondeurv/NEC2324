using Accord.Statistics.Analysis;
using CommandLine;
using SupportVectorMachines.BP;
using SupportVectorMachines.BP.Extensions;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;
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

        IModel model = default;
        int[] predicted = default;
        (NDArray X, NDArray Y) test = default;
        if (opt.ModelFile != null)
        {
            test = dataset.ToNDArrays();
            model = KerasApi.keras.models.load_model(opt.ModelFile);
            predicted = model
                .predict(test.X)
                .numpy()
                .ToArray<float>()
                .Select(p => p > opt.Threshold ? 1 : 0)
                .ToArray();
        }
        else
        {
            var activationFunction = GetActivationFunction(opt.ActivationFunction);
            var firstLayer = true;
            var layers = opt.Layers
                .Split(":")
                .Select(u =>
                {
                    var layer = new Dense(new DenseArgs
                    {
                        Units = int.Parse(u),
                        Activation = activationFunction,
                        InputShape = firstLayer ? new Shape(trainDataset.Data.Keys.Count() - 1) : default,
                    });

                    if (firstLayer) firstLayer = false;
                    return layer;
                }).ToList<ILayer>();

            model = new Sequential(new SequentialArgs
            {
                Layers = layers
            });

            model.compile(optimizer: new SGD(opt.LearningRate, opt.Momentum), loss: new MeanSquaredError(),
                metrics: new[] { "accuracy" });
            var train = trainDataset.ToNDArrays();
            model.fit(train.X, train.Y, epochs: opt.Epochs, batch_size: 32, verbose: 1);
            test = testDataset.ToNDArrays();
            predicted = model
                .predict(test.X)
                .numpy()
                .ToArray<float>()
                .Select(p => p > opt.Threshold ? 1 : 0)
                .ToArray();
        }
        
        var modelFile = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-bp-model";
        model.save(modelFile);
        Console.WriteLine($"Model saved to {modelFile}");
        
         var confusionMatrix = new ConfusionMatrix(predicted,
             test.Y.Select(o => (int)o).ToArray());
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
        
         var roc = new ReceiverOperatingCharacteristic(test.Y.numpy().ToArray<float>().Select(v => (double)v).ToArray(),
             predicted.Select(p => (double)p).ToArray());
         roc.Compute(1000);
        
         Console.WriteLine($"AUC: {roc.Area}");
         if (opt.ExportPlots)
         {
             var features = (opt.PlotFeatures?.Split(':') ?? dataset.Data.Keys.Take(2)).ToArray();
             testDataset.Data.TryGetValue(features[0], out var featureX);
             testDataset.Data.TryGetValue(features[1], out var featureY);
        
             var fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-bp-expected.png";
             PlotExporter.ExportScatterPlot($"Actual {dataset.Data.Keys.Last()}", features, featureX, featureY,
                 test.Y.numpy().ToArray<float>().Select(v => (int)v).ToArray(),
                 fileName);
             Console.WriteLine($"Exported plot to {fileName}");
        
             fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-bp-predicted.png";
             PlotExporter.ExportScatterPlot($"Predicted {dataset.Data.Keys.Last()}", features, featureX, featureY,
                 predicted,
                 fileName);
             Console.WriteLine($"Exported plot to {fileName}");
        
             fileName = $"{Path.GetFileNameWithoutExtension(opt.DatasetFile)}-bp-roc.png";
             var points = roc.Points.Select(p =>
                 (p.FalsePositiveRate, (double)p.TruePositives / (p.TruePositives + p.FalseNegatives)));
             PlotExporter.ExportRoc(points, fileName);
             Console.WriteLine($"Exported ROC plot to {fileName}");
         }
    });
    
    static Activation GetActivationFunction(string activationFunction)
    {
        return activationFunction switch
        {
            "sigmoid" => KerasApi.keras.activations.Sigmoid,
            "tanh" => KerasApi.keras.activations.Tanh,
            "relu" => KerasApi.keras.activations.Relu,
            "linear" => KerasApi.keras.activations.Linear,
            _ => throw new ArgumentOutOfRangeException(nameof(activationFunction), activationFunction,
                "Invalid activation function.")
        };
    }