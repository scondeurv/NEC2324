using BackPropagation.ActivationFunctions;
using BackPropagation.Exceptions;
using BackPropagation.Extensions;
using BackPropagation.Scaling;
using Microsoft.Extensions.Logging;

namespace BackPropagation;

public sealed class MyNeuralNetwork
{
    private ILogger Logger { get; }
    private string[] Features { get; }
    private double[][] Data { get; }
    private int L { get; } //Number of layers
    private int[] N { get; } //Units per layer
    private double[][] H { get; } // Fields
    private double[][] Xi { get; } // Activations
    private double[][,] W { get; } // Weights
    private double[][] Theta { get; } //Thresholds
    private double[][] Delta { get; } // Error propagation
    private double[][,] D_W { get; } //Weight changes
    private double[][] D_Theta { get; } //Threshold changes
    private double[][,] D_W_Prev { get; } //Weight previous changes
    private double[][] D_Theta_Prev { get; } //Threshold previous changes
    private double[] Y { get; set; } //Predictions
    private double[] Z { get; set; } //Expected values
    private double E { get; set; } //Quadratic error
    private IActivationFunction Fact { get; }
    private int Epochs { get; }
    private double ValidationPercentage { get; }

    public MyNeuralNetwork(ILogger logger, string[] features, double[][] data, int[] unitsPerLayer,
        ActivationFunctionType fact, int epochs, double validationPercentage)
    {
        Logger = logger;
        Features = features;
        Data = data;

        L = unitsPerLayer.Length;
        N = unitsPerLayer;
        H = new double[L][];
        Xi = new double[L][];
        Xi[0] = new double[N[0]];
        W = new double[L][,];
        D_W = new double[L][,];
        D_W_Prev = new double[L][,];
        Theta = new double[L][];
        D_Theta = new double[L][];
        D_Theta_Prev = new double[L][];
        Delta = new double[L][];
        for (var layer = 0; layer < L; layer++)
        {
            Xi[layer] = new double[N[layer]];
            H[layer] = new double[N[layer]];
            Delta[layer] = new double[N[layer]];
            Theta[layer] = new double[N[layer]];
            D_Theta[layer] = new double[N[layer]];
            D_Theta_Prev[layer] = new double[N[layer]];
            if (layer > 0)
            {
                W[layer] = new double[N[layer], N[layer - 1]];
                D_W[layer] = new double[N[layer], N[layer - 1]];
                D_W_Prev[layer] = new double[N[layer], N[layer - 1]];
            }
        }

        Y = new double[data.Length];
        Z = new double[data.Length];

        var factory = new ActivationFunctionFactory();
        Fact = factory.Create(fact);

        Epochs = epochs;
        ValidationPercentage = validationPercentage;
    }

    public async Task Fit(double learningRate, double momentum,
        IReadOnlyDictionary<string, IScalingMethod>? scalingPerFeature = null,
        CancellationToken? cancellationToken = null)
    {
        double[][] data;
        if (scalingPerFeature.Any())
        {
            data = await ScaleData(Data, scalingPerFeature, cancellationToken);
        }
        else
        {
            data = Data;
        }

        await Task.WhenAll(
            InitializeWeights((0, 1), cancellationToken),
            InitializeThresholds((0, 1), cancellationToken));

        var datasets = await SplitDataSet(data, ValidationPercentage, cancellationToken);
        var plotData = new List<(double, double)>(Epochs);
        for (var epoch = 1; epoch <= Epochs; epoch++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            
            var rand = new Random();

            for (var i = 0; i < datasets.TrainingSet.Length; i++)
            {
                var pattern = rand.Next(0, datasets.TrainingSet.Length);
                await InitXi(datasets.TrainingSet, pattern, cancellationToken);
                await FeedForward(cancellationToken);
                await BackPropagate(ox: Xi[^1][0], z: datasets.TrainingSet[pattern][^1], cancellationToken);
                await Task.WhenAll(
                    UpdateWeights(learningRate, momentum, cancellationToken),
                    UpdateThresholds(learningRate, momentum, cancellationToken));
            }

            Y = new double[data.Length];
            Z = new double[data.Length];
            E = 0;
            for (var pattern = 0; pattern < datasets.TrainingSet.Length; pattern++)
            {
                await InitXi(datasets.TrainingSet, pattern, cancellationToken);
                await FeedForward(cancellationToken);

                if (scalingPerFeature.TryGetValue(Features[^1], out var outputScalingMethod))
                {
                    Y[pattern] = (await outputScalingMethod.Descale(new [] { Xi[^1][0] }, cancellationToken))[0];
                    Z[pattern] = (await outputScalingMethod.Descale(new [] { datasets.TrainingSet[pattern][^1] }, cancellationToken))[0];
                }
                else
                {
                    Y[pattern] = Xi[^1][0];
                    Z[pattern] = datasets.TrainingSet[pattern][^1];
                }
                
                E += Math.Abs((Y[pattern] - Z[pattern])/Z[pattern]);
            }

            var trainingError = 100 * E / datasets.TrainingSet.Length;
            plotData.Add((epoch, trainingError));
            Logger.LogInformation($"Training error: {trainingError}");
        }

        var plotExporter = new PlotExporter();
        
        plotExporter.ExportLinear(
            $"MAPE vs Epoch (\u03B7: {learningRate:F4}, \u03B1: {momentum:F4})", 
            "Epoch", 
            "MAPE",
            new Dictionary<string, (double X, double Y)[]> {
                { string.Empty, plotData.ToArray()}
            },
            $"MapeVsEpoch-{DateTime.Now:yyyyMMddhhmmss}");
        
        Y = new double[data.Length];
        Z = new double[data.Length];
        E = 0;
        var scatterData = new List<(double, double)>(datasets.ValidationSet.Length);
        for (var pattern = 0; pattern < datasets.ValidationSet.Length; pattern++)
        {
            await InitXi(datasets.ValidationSet, pattern, cancellationToken);
            await FeedForward(cancellationToken);
            
            if (scalingPerFeature.TryGetValue(Features[^1], out var outputScalingMethod))
            {
                Y[pattern] = (await outputScalingMethod.Descale(new [] { Xi[^1][0] }, cancellationToken))[0];
                Z[pattern] = (await outputScalingMethod.Descale(new [] { datasets.ValidationSet[pattern][^1] }, cancellationToken))[0];
            }
            else
            {
                Y[pattern] = Xi[^1][0];
                Z[pattern] = datasets.ValidationSet[pattern][^1];
            }
            
            scatterData.Add((Y[pattern], Z[pattern]));
            E += Math.Abs((Y[pattern] - Z[pattern])/Z[pattern]);
        }
        plotExporter.ExportScatter(
            $"Actual Vs Prediction (\u03B7: {learningRate:F4}, \u03B1: {momentum:F4})", 
            "Prediction", 
            "Actual",
            scatterData.ToArray(),
            $"ActualVsPrediction-{DateTime.Now:yyyyMMddhhmmss}");
        var validationError = 100 * E / datasets.ValidationSet.Length;

        Logger.LogInformation($"Validation error: {validationError}");
    }

    private Task FeedForward(CancellationToken? cancellationToken)
    {
        for (var layer = 1; layer < L; layer++)
        {
            for (var i = 0; i < N[layer]; i++)
            {
                double sum = 0;
                for (var j = 0; j < N[layer - 1]; j++)
                {
                    cancellationToken?.ThrowIfCancellationRequested();
                    sum += W[layer][i, j] * Xi[layer - 1][j];
                }

                var theta = Theta[layer][i];
                var h = sum - theta;
                H[layer][i] = h;
                Xi[layer][i] = Fact.Eval(h);
            }
        }

        return Task.CompletedTask;
    }

    private Task BackPropagate(double ox, double z, CancellationToken? cancellationToken)
    {
        for (var i = 0; i < N[L - 1]; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();

            Delta[L - 1][i] = Fact.Derivative(H[L - 1][i]) * (ox - z);
        }

        for (var layer = L - 1; layer > 1; layer--)
        {
            cancellationToken?.ThrowIfCancellationRequested();

            var sum = 0.0;
            for (var j = 0; j < N[layer - 1]; j++)
            {
                for (var i = 0; i < N[layer]; i++)
                {
                    sum += Delta[layer][i] * W[layer][i, j];
                }

                var gPrime = Fact.Derivative(H[layer - 1][j]);
                Delta[layer - 1][j] = gPrime * sum;
            }
        }

        return Task.CompletedTask;
    }

    private Task UpdateWeights(double learningRate, double momentum, CancellationToken? cancellationToken)
    {
        for (var layer = L - 1; layer > 0; layer--)
        {
            for (var j = 0; j < N[layer - 1]; j++)
            {
                for (var i = 0; i < N[layer]; i++)
                {
                    cancellationToken?.ThrowIfCancellationRequested();

                    D_W[layer][i, j] = (-learningRate * Delta[layer][i] * Xi[layer - 1][j]) +
                                       (momentum * D_W_Prev[layer][i, j]);
                    D_W_Prev[layer][i, j] = D_W[layer][i, j];
                    W[layer][i, j] += D_W[layer][i, j];
                }
            }
        }

        return Task.CompletedTask;
    }

    private Task UpdateThresholds(double learningRate, double momentum, CancellationToken? cancellationToken)
    {
        for (var layer = L - 1; layer > 0; layer--)
        {
            for (var i = 0; i < N[layer]; i++)
            {
                cancellationToken?.ThrowIfCancellationRequested();

                D_Theta[layer][i] = (learningRate * Delta[layer][i]) + (momentum * D_Theta_Prev[layer][i]);
                D_Theta_Prev[layer][i] = D_Theta[layer][i];
                Theta[layer][i] += D_Theta[layer][i];
            }
        }

        return Task.CompletedTask;
    }

    private Task InitXi(double[][] data, int pattern, CancellationToken? cancellationToken)
    {
        for (var feature = 0; feature < N[0]; feature++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            Xi[0][feature] = data[pattern][feature];
        }

        return Task.CompletedTask;
    }

    private async Task<(double[][] TrainingSet, double[][] ValidationSet)> SplitDataSet(
        double[][] data,
        double validationPercentage,
        CancellationToken? cancellationToken)
    {
        if (validationPercentage == 0)
        {
            return (data, Array.Empty<double[]>());
        }

        var shuffledData = await Shuffle(data, cancellationToken);
        var trainingSize = (int)(shuffledData.Length * (1 - validationPercentage));
        var validationSize = shuffledData.Length - trainingSize;

        var trainingSet = new double[trainingSize][];
        var validationSet = new double[validationSize][];

        for (var row = 0; row < trainingSize; row++)
        {
            trainingSet[row] = data[row];
        }

        for (var row = trainingSize; row < shuffledData.Length; row++)
        {
            validationSet[row - trainingSize] = data[row];
        }

        return (trainingSet, validationSet);
    }

    private Task<double[][]> Shuffle(double[][] data, CancellationToken? cancellationToken)
    {
        var random = new Random();
        for (var row = 0; row < data.Length; row++)
        {
            var newRow = random.Next(0, data.Length - 1);
            (data[newRow], data[row]) = (data[row], data[newRow]);
        }

        return Task.FromResult(data);
    }

    private Task InitializeWeights((double Min, double Max) range, CancellationToken? cancellationToken)
    {
        var rand = new Random();
        foreach (var w in W)
        {
            if (w is null) continue;
            for (var i = 0; i < w.GetLength(0); i++)
            {
                for (var j = 0; j < w.GetLength(1); j++)
                {
                    cancellationToken?.ThrowIfCancellationRequested();
                    w[i, j] = rand.NextDouble(range.Min, range.Max);
                }
            }
        }
        
        return Task.CompletedTask;
    }

    private Task InitializeThresholds((double Min, double Max) range, CancellationToken? cancellationToken)
    {
        var rand = new Random();
        for (var i = 0; i < Theta.Length; i++)
        {
            for (var j = 0; j < Theta[i].Length; j++)
            {
                cancellationToken?.ThrowIfCancellationRequested();
                Theta[i][j] = rand.NextDouble(range.Min, range.Max);
            }
        }
        
        return Task.CompletedTask;
    }

    private async Task<double[][]> ScaleData(double[][] data,
        IReadOnlyDictionary<string, IScalingMethod> scalingMethodPerFeature,
        CancellationToken? cancellationToken = null)
    {
        var features = scalingMethodPerFeature.Keys.ToList();
        var scalingTasks = new List<Task<double[]>>();
        var featureIndexes = new List<int>();
        foreach (var feature in features)
        {
            cancellationToken?.ThrowIfCancellationRequested();

            if (Features.Contains(feature))
            {
                var col = features.IndexOf(feature);
                featureIndexes.Add(col);
                var featureData = new double[data.Length];
                for (var row = 0; row < data.Length; row++)
                {
                    featureData[row] = data[row][col];
                }

                var scalingMethod = scalingMethodPerFeature[feature];
                scalingTasks.Add(scalingMethod.Scale(featureData, cancellationToken));
            }
            else
            {
                throw new FeatureNotFoundException(feature);
            }
        }

        var results = await Task.WhenAll(scalingTasks);
        var scaledData = new double[data.Length][];
        for (var row = 0; row < data.Length; row++)
        {
            scaledData[row] = new double[data[0].Length];

            for (var col = 0; col < data[0].Length; col++)
            {
                if (featureIndexes.Contains(col))
                {
                    scaledData[row][col] = results[col][row];
                }
                else
                {
                    scaledData[row][col] = data[row][col];
                }
            }
        }

        return scaledData;
    }
}