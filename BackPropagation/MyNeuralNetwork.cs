using BackPropagation.ActivationFunctions;
using BackPropagation.Extensions;
using Microsoft.Extensions.Logging;

namespace BackPropagation;

public sealed class MyNeuralNetwork
{
    private const string MseError = "MSE";
    private const string MapeError = "MAPE";
    private ILogger Logger { get; }
    private IActivationFunction Fact { get; }
    private int Epochs { get; }
    private double ValidationPercentage { get; }
    private int L { get; } //Number of layers
    private int[] N { get; } //Units per layer
    private double LearningRate { get; }
    private double Momentum { get; }
    private double[][] H { get; set; } // Fields
    private double[][] Xi { get; set; } // Activations
    private double[][,] W { get; set; } // Weights
    private double[][] Theta { get; set; } //Thresholds
    private double[][] Delta { get; set; } // Error propagation
    private double[][,] D_W { get; set; } //Weight changes
    private double[][] D_Theta { get; set; } //Threshold changes
    private double[][,] D_W_Prev { get; set; } //Weight previous changes
    private double[][] D_Theta_Prev { get; set; } //Threshold previous changes
    private double[] TrainingErrors { get; set; }
    private double[] ValidationErrors { get; set; }

    public MyNeuralNetwork(ILogger logger, int[] unitsPerLayer,
        ActivationFunctionType fact, int epochs, double validationPercentage, double learningRate, double momentum)
    {
        Logger = logger;

        L = unitsPerLayer.Length;
        N = unitsPerLayer;
        Epochs = epochs;
        ValidationPercentage = validationPercentage;
        LearningRate = learningRate;
        Momentum = momentum;

        var factory = new ActivationFunctionFactory();
        Fact = factory.Create(fact);
    }

    public async Task Fit(double[][] data, string[] features, CancellationToken? cancellationToken = null)
    {
        Logger.LogInformation("Initializing training...");
        Init();
        await Task.WhenAll(
            InitializeWeights((0, 1), cancellationToken),
            InitializeThresholds((0, 1), cancellationToken));

        var datasets = await SplitDataSet(data, ValidationPercentage, cancellationToken);

        Logger.LogInformation("Training...");
        for (var epoch = 0; epoch < Epochs; epoch++)
        {
            cancellationToken?.ThrowIfCancellationRequested();

            Logger.LogInformation($"Epoch {epoch + 1}");
            var rand = new Random();
            foreach (var t in datasets.TrainingSet)
            {
                var pattern = rand.Next(0, datasets.TrainingSet.Length);
                await InitXi(datasets.TrainingSet, pattern, cancellationToken);
                await FeedForward(cancellationToken);
                await BackPropagation(y: Xi[^1][0], z: datasets.TrainingSet[pattern][^1], cancellationToken);
                await Task.WhenAll(
                    UpdateWeights(LearningRate, Momentum, cancellationToken),
                    UpdateThresholds(LearningRate, Momentum, cancellationToken));
            }

            TrainingErrors[epoch] = await CalculateError(datasets.TrainingSet, MapeError, cancellationToken);
            ValidationErrors[epoch] =
                await CalculateError(datasets.ValidationSet, MapeError, cancellationToken);
        }

        Logger.LogInformation("Training ended");
    }

    public (double[] TrainingErrors, double[] ValidationErrors) LossEpochs() => (TrainingErrors, ValidationErrors);

    public async Task<double[]> Predict(double[][] data, CancellationToken? cancellationToken = null)
    {
        var predictions = new List<double>();
        Logger.LogInformation("Working...");
        for (var pattern = 0; pattern < data.Length; pattern++)
        {
            Logger.LogInformation($"Pattern {pattern + 1}");
            await InitXi(data, pattern, cancellationToken);
            var prediction = await FeedForward(cancellationToken);
            predictions.Add(prediction);
        }

        return predictions.ToArray();
    }

    private void Init()
    {
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

        TrainingErrors = new double[Epochs];
        ValidationErrors = new double[Epochs];
    }

    private Task<double> FeedForward(CancellationToken? cancellationToken)
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

        return Task.FromResult(Xi[^1][0]);
    }

    private Task BackPropagation(double y, double z, CancellationToken? cancellationToken)
    {
        for (var i = 0; i < N[L - 1]; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();

            Delta[L - 1][i] = Fact.Derivative(H[L - 1][i]) * (y - z);
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

        for (var i = 1; i < Xi.Length; i++)
        {
            for (var j = 0; j < Xi[i].Length; j++)
            {
                Xi[i][j] = 0;
            }
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
            cancellationToken?.ThrowIfCancellationRequested();
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

    private async Task<double> CalculateError(double[][] data, string errorType, CancellationToken? cancellationToken)
    {
        var predictions = await Predict(data, cancellationToken);
        
        var sum = 0.0;
        for (var i = 0; i < data.Length; i++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            var z = data[i][^1];
            var y = predictions[i];
            var e = errorType == MseError ? Math.Pow(y - z, 2) : Math.Abs(z - y);
            sum += e;
        }
        
        var error = (errorType == MseError ? 0.5 : 100.00/data.Length)  * sum;
        return error;
    }
}