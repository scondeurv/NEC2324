using BackPropagation.Scaling;
using Microsoft.Extensions.Logging;

namespace BackPropagation;

public sealed class NeuralNetwork
{
    private ILogger Logger { get; init; }
    private string[] Features { get; init; }
    private double[][] Data { get; init; }
    private string OutputFeature { get; init; }
    private int L { get; init; } //Number of layers
    private int[] N { get; init; } //Units per layer
    private double[][] H { get; init; } // Fields
    private double[][] Xi { get; init; } // Activations
    private double[][,] W { get; init; } // Weights
    private double[][] Theta { get; init; } //Thresholds
    private double[][] Delta { get; init; } // Error propagation
    private double[][,] D_W { get; init; } //Weight changes
    private double[][] D_Theta { get; init; } //Threshold changes
    private double[][,] D_W_Prev { get; init; } //Weight previous changes
    private double[][] D_Theta_Prev { get; init; } //Threshold previous changes
    private ActivationFunction Fact { get; init; }

    public NeuralNetwork(ILogger logger, string[] features, double[][] data, string outputFeature, int[] unitsPerLayer,
        ActivationFunction fact)
    {
        Features = features;
        Logger = logger;
        L = unitsPerLayer.Length;
        N = unitsPerLayer;

        H = new double[L][];
        Xi = new double[L][];
        W = new double[L][,];
        Theta = new double[L][];
        Delta = new double[L][];
        D_W = new double[L][,];
        D_Theta = new double[L][];
        D_W_Prev = new double[L][,];
        D_Theta_Prev = new double[L][];
        for (var layer = 0; layer < L; layer++)
        {
            H[layer] = new double[N[layer]];
            Xi[layer] = new double[N[layer]];
            W[layer] = new double[data.Length, N[layer]];
            Theta[layer] = new double[N[layer]];
            Delta[layer] = new double[N[layer]];
            D_W[layer] = new double[data.Length, N[layer]];
            D_Theta[layer] = new double[N[layer]];
            D_W_Prev[layer] = new double[data.Length, N[layer]];
            D_Theta_Prev[layer] = new double[N[layer]];
        }

        OutputFeature = outputFeature;
        Data = data;
        Fact = fact;
    }

    public Task Train(int epochs, float trainingDataPercentage, float testDataPercentage, double learningRate, double momentum, IScalingMethod scalingMethod)
    {
    }
    
}