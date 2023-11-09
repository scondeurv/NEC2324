using BackPropagation.ActivationFunctions;
using BackPropagation.Exceptions;
using BackPropagation.Extensions;
using BackPropagation.Scaling;
using Microsoft.Extensions.Logging;

namespace BackPropagation;

public sealed class NeuralNetwork
{
    private ILogger Logger { get; }
    private ISet<string> Features { get; }
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
    private IActivationFunction Fact { get; }

    public NeuralNetwork(ILogger logger, ISet<string> features, double[][] data, int[] unitsPerLayer, ActivationFunctionType fact)
    {
        Features = features;
        Logger = logger;
        L = unitsPerLayer.Length;
        N = unitsPerLayer;

        H = new double[L-1][];
        Xi = new double[L][];
        Xi[0] = new double[Data[0].Length];
        W = new double[L-1][,];
        Theta = new double[L-1][];
        Delta = new double[L-1][];
        D_W = new double[L-1][,];
        D_Theta = new double[L-1][];
        D_W_Prev = new double[L-1][,];
        D_Theta_Prev = new double[L-1][];
        for (var layer = 0; layer < L - 1; layer++)
        {
            H[layer] = new double[N[layer]];
            Xi[layer+1] = new double[N[layer+1]];
            W[layer] = new double[data.Length, N[layer]];
            Theta[layer] = new double[N[layer]];
            Delta[layer] = new double[N[layer]];
            D_W[layer] = new double[data.Length, N[layer]];
            D_Theta[layer] = new double[N[layer]];
            D_W_Prev[layer] = new double[data.Length, N[layer]];
            D_Theta_Prev[layer] = new double[N[layer]];
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
            InitializeWeights(cancellationToken), 
            InitializeThresholds(cancellationToken));

        var datasets = await SplitDataSet(trainingDataPercentage, cancellationToken);
        
        for (var epoch = 1; epoch <= numberOfEpochs; epoch++)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            
            Logger.LogInformation($"Running Epoch {epoch}...");

            for (var pattern = 0; pattern < datasets.TrainingSet[0].Length; pattern++)
            {
                Logger.LogInformation($"Feed forward {epoch}.{pattern}...");
                await InitXi(datasets.TrainingSet, pattern, cancellationToken);   
                for (var layer = 0; layer < L; layer++)
                {
                    Logger.LogInformation($"On layer {layer}...");
                    for (var unit = 0; unit < N[layer]; unit++)
                    {
                        double sum = 0;
                        for(var i = 0; i < Xi[layer].Length; i++)
                        {
                            sum += W[layer][unit, i] * Xi[layer][i];
                        }

                        var theta = Theta[layer][unit];
                        var h = sum - theta;
                        H[layer][unit] = h;
                        Xi[layer + 1][unit] = Fact.Eval(h);
                    }
                }
                Logger.LogInformation($"Feed forward {epoch}.{pattern} ended");
                Logger.LogInformation($"Back propagation {epoch}.{pattern}...");
                var ox = Xi[Xi.Length][0];
                var z = datasets.TrainingSet[^1][pattern];
                for (var unit = 0; unit < Delta[L - 1].Length; unit++)
                {
                    Delta[L - 1][unit] = Fact.Derivative(H[L - 1][unit]) * (ox - z);
                }
                
                for (var layer = L - 2; layer >= 0; layer--)
                {
                    for (var unit = 0; unit < Delta[layer].Length; unit++)
                    {
                        var sum = 0.0;
                        for (var i = 0; i < Delta[layer + 1].Length; i++)
                        {
                            sum += Delta[layer + 1][i] * W[layer + 1][unit, i];
                        }

                        var gPrime = Fact.Derivative(H[layer + 1][unit]);
                        Delta[layer][unit] = gPrime * sum;
                    }
                }
                Logger.LogInformation($"Back propagation {epoch}.{pattern} ended");
            }
        }
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
    
    private async Task<(double[][] TrainingSet, double[][] TestSet)> SplitDataSet(double trainingSetPercentage, CancellationToken? cancellationToken)
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

    private Task InitializeWeights(CancellationToken? cancellationToken)
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
                    w[i, j] = rand.NextDouble(0.0, 1.0);
                }
            }
        }
        
        Logger.LogInformation("Weights initialized.");
        return Task.CompletedTask;
    }

    private Task InitializeThresholds(CancellationToken? cancellationToken)
    {        
        Logger.LogInformation("Initializing thresholds...");
        var rand = new Random();

        for (var i = 0; i < Theta.Length; i++)
        {
            for (var j = 0; j < Theta[i].Length; j++)
            {
                cancellationToken?.ThrowIfCancellationRequested();
                Theta[i][j] = rand.NextDouble(0.0, 1.0);
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