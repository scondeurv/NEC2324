using BackPropagation.ActivationFunctions;
using BackPropagation.Exceptions;
using BackPropagation.Extensions;
using BackPropagation.Scaling;
using Microsoft.Extensions.Logging;

namespace BackPropagation;

public sealed class NeuralNetwork
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
    private double[] Ox { get; } //Predictions
    private double[] Z { get; } //Expected values
    private double[] E { get; } //Quadratic error
    private IActivationFunction Fact { get; }

    public NeuralNetwork(ILogger logger, string[] features, double[][] data, int[] unitsPerLayer,
        ActivationFunctionType fact)
    {
        Features = features;
        Logger = logger;
        L = unitsPerLayer.Length;
        N = unitsPerLayer;

        Ox = new double[data[0].Length];
        H = new double[L - 1][];
        Xi = new double[L][];
        Xi[0] = new double[unitsPerLayer[0]];
        W = new double[L - 1][,];
        Theta = new double[L - 1][];
        Delta = new double[L - 1][];
        D_W = new double[L - 1][,];
        D_Theta = new double[L - 1][];
        D_W_Prev = new double[L - 1][,];
        D_Theta_Prev = new double[L - 1][];
        for (var layer = 1; layer < L; layer++)
        {
            H[layer - 1] = new double[N[layer]];
            Xi[layer] = new double[N[layer]];
            W[layer - 1] = new double[N[layer - 1], N[layer]];
            Theta[layer - 1] = new double[N[layer]];
            Delta[layer - 1] = new double[N[layer]];
            D_W[layer - 1] = new double[data.Length, N[layer]];
            D_Theta[layer - 1] = new double[N[layer]];
            D_W_Prev[layer - 1] = new double[data.Length, N[layer]];
            D_Theta_Prev[layer - 1] = new double[N[layer]];
        }

        Data = data;

        var factory = new ActivationFunctionFactory();
        Fact = factory.Create(fact);
    }

    public async Task Train(int numberOfEpochs, int trainingDataPercentage,
        double learningRate, double momentum, IReadOnlyDictionary<string, IScalingMethod>? scalingPerFeature = null,
        CancellationToken? cancellationToken = null)
    {
        double[][] data;
        if (scalingPerFeature is null)
        {
            data = await CloneData(cancellationToken);
        }
        else
        {
            Logger.LogInformation("Scaling features...");
            data = await ScaleData(Data, scalingPerFeature, cancellationToken);
            Logger.LogInformation("Features scaled.");
        }

        await Task.WhenAll(
            InitializeWeights((-1, 1),cancellationToken),
            InitializeThresholds((-1, 1), cancellationToken));

        var datasets = await SplitDataSet(trainingDataPercentage, cancellationToken);

        for (var epoch = 1; epoch <= numberOfEpochs; epoch++)
        {
            cancellationToken?.ThrowIfCancellationRequested();

            Logger.LogInformation($"Running Epoch {epoch}...");

            for (var pattern = 0; pattern < datasets.TrainingSet[0].Length; pattern++)
            {
                Logger.LogInformation($"Feed forward {epoch}.{pattern}...");
                
                await InitXi(datasets.TrainingSet, pattern, cancellationToken);
                await FeedForward(cancellationToken);
                
                Logger.LogInformation($"Feed forward {epoch}.{pattern} ended");

                Logger.LogInformation($"Back propagation {epoch}.{pattern}...");
                
                Ox[pattern] = Xi[Xi.Length][0];
                Z[pattern] = datasets.TrainingSet[^1][pattern];
                E[pattern] = 0.5 * Math.Pow(Ox[pattern] * Z[pattern], 2);
                
                await BackPropagate(Ox[pattern], Z[pattern], cancellationToken);
                
                Logger.LogInformation($"Back propagation ended: {epoch}.{pattern} ");

                Logger.LogInformation($"Updating weights and thresholds: {epoch}.{pattern}...");
                
                await Task.WhenAll(
                    UpdateWeights(learningRate, momentum, cancellationToken),
                    UpdateThresholds(learningRate, momentum, cancellationToken));
                
                Logger.LogInformation($"Weight and thresholds updated: {epoch}.{pattern}...");
            }

            var quadraticError = E.Sum(e => e);
            
            Logger.LogInformation($"Quadratic error: {quadraticError}");
            Logger.LogInformation($"Epoch ended: {epoch}");
        }
    }

    private Task FeedForward(CancellationToken? cancellationToken)
    {
        for (var layer = 1; layer < L; layer++)
        {
            Logger.LogInformation($"On layer {layer}...");
            for (var j = 0; j < N[layer]; j++)
            {
                cancellationToken?.ThrowIfCancellationRequested();

                double sum = 0;
                for (var i = 0; i < N[layer - 1]; i++)
                {
                    sum += W[layer][i, j] * Xi[layer - 1][i];
                }

                var theta = Theta[layer][j];
                var h = sum - theta;
                H[layer][j] = h;
                Xi[layer][j] = Fact.Eval(h);
            }
        }

        return Task.CompletedTask;
    }

    private Task BackPropagate(double ox, double z, CancellationToken? cancellationToken)
    {
        for (var i = 0; i < Delta[L - 1].Length; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();

            Delta[L - 1][i] = Fact.Derivative(H[L - 1][i]) * (ox - z);
        }

        for (var layer = L - 2; layer >= 0; layer--)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            for (var j = 0; j < N[layer]; j++)
            {
                var sum = 0.0;
                for (var i = 0; i < N[layer + 1]; i++)
                {
                    sum += Delta[layer + 1][i] * W[layer + 1][i, j];
                }

                var gPrime = Fact.Derivative(H[layer][j]);
                Delta[layer][j] = gPrime * sum;
            }
        }

        return Task.CompletedTask;
    }

    private Task UpdateWeights(double learningRate, double momentum, CancellationToken? cancellationToken)
    {
        for (var layer = L - 2; layer >= 0; layer--)
        {
            for (var i = 0; i < N[layer - 1]; i++)
            {
                for (var j = 0; j < N[layer]; j++)
                {
                    cancellationToken?.ThrowIfCancellationRequested();

                    D_W[layer][i, j] = -learningRate * Delta[layer][i] * Xi[layer - 1][j] +
                                       momentum * D_W_Prev[layer][i, j];
                    D_W_Prev[layer][i, j] = D_W[layer][i, j];
                    W[layer][i, j] += D_W[layer][i, j];
                }
            }
        }

        return Task.CompletedTask;
    }

    private Task UpdateThresholds(double learningRate, double momentum, CancellationToken? cancellationToken)
    {
        for (var layer = L - 2; layer >= 0; layer--)
        {
            for (var i = 0; i < N[layer - 1]; i++)
            {
                cancellationToken?.ThrowIfCancellationRequested();

                D_Theta[layer][i] = learningRate * Delta[layer][i] + momentum * D_Theta_Prev[layer][i];
                D_Theta_Prev[layer][i] = D_Theta[layer][i];
                Theta[layer][i] += D_Theta[layer][i];
            }
        }

        return Task.CompletedTask;
    }

    private Task InitXi(double[][] data, int pattern, CancellationToken? cancellationToken)
    {
        for (var i = 0; i < Xi[0].Length; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            Xi[0][i] = data[i][pattern];
        }

        return Task.CompletedTask;
    }

    private async Task<(double[][] TrainingSet, double[][] TestSet)> SplitDataSet(double trainingSetPercentage,
        CancellationToken? cancellationToken)
    {
        var shuffledData = await Shuffle(Data, cancellationToken);
        var trainingSize = (int)(shuffledData.Length * trainingSetPercentage / 100);
        var testSize = shuffledData.Length - trainingSize;

        var trainingSet = new double[trainingSize][];
        var testSet = new double[testSize][];

        Array.Copy(shuffledData, 0, trainingSet, 0, trainingSize);
        Array.Copy(shuffledData, trainingSize, testSet, 0, testSize);

        return (trainingSet, testSet);
    }

    private Task<double[][]> Shuffle(double[][] data, CancellationToken? cancellationToken)
    {
        var random = new Random();
        var swapped = new SortedSet<int>();
        for (var col = 0; col < data[0].Length; col++)
        {
            for (var row = 0; row < data.Length; row++)
            {
                int? newCol = null;
                while (newCol is null)
                {
                    cancellationToken?.ThrowIfCancellationRequested();
                    var rand = random.Next(0, col);
                    newCol = !swapped.Contains(rand) && rand != col ? rand : null;
                }

                (data[row][col], data[row][newCol.Value]) = (data[row][newCol.Value], data[row][col]);
            }
        }

        return Task.FromResult(data);
    }

    private Task InitializeWeights((double Min, double Max) range, CancellationToken? cancellationToken)
    {
        Logger.LogInformation("Initializing weights...");
        var rand = new Random();
        foreach (var w in W)
        {
            for (var i = 0; i < w.GetLength(0); i++)
            {
                for (var j = 0; j < w.GetLength(1); j++)
                {
                    cancellationToken?.ThrowIfCancellationRequested();
                    w[i, j] = rand.NextDouble(range.Min, range.Max);
                }
            }
        }

        Logger.LogInformation("Weights initialized.");
        return Task.CompletedTask;
    }

    private Task InitializeThresholds((double Min, double Max) range, CancellationToken? cancellationToken)
    {
        Logger.LogInformation("Initializing thresholds...");
        var rand = new Random();

        for (var i = 0; i < Theta.Length; i++)
        {
            for (var j = 0; j < Theta[i].Length; j++)
            {
                cancellationToken?.ThrowIfCancellationRequested();
                Theta[i][j] = rand.NextDouble(range.Min, range.Max);
            }
        }

        Logger.LogInformation("Thresholds initialized.");
        return Task.CompletedTask;
    }

    private Task<double[][]> CloneData(CancellationToken? cancellationToken)
    {
        var data = new double[Data.Length][];
        for (var i = 0; i < Data.Length; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            data[i] = new double[Data[i].Length];
            Array.Copy(Data[i], data[i], Data[i].Length);
        }

        return Task.FromResult(data);
    }

    private async Task<double[][]> ScaleData(double[][] data,
        IReadOnlyDictionary<string, IScalingMethod> scalingMethodPerFeature,
        CancellationToken? cancellationToken = null)
    {
        var features = scalingMethodPerFeature.Keys.ToList();
        var scalingTasks = new List<Task<double[]>>();
        foreach (var feature in features)
        {
            cancellationToken?.ThrowIfCancellationRequested();

            if (Features.Contains(feature))
            {
                var featureIndex = features.IndexOf(feature);
                var featureData = data[featureIndex];
                var scalingMethod = scalingMethodPerFeature[feature];
                scalingTasks.Add(scalingMethod.Scale(featureData, cancellationToken));
            }
            else
            {
                throw new FeatureNotFoundException(feature);
            }
        }

        var scaledData = await Task.WhenAll(scalingTasks);
        return scaledData;
    }
}