using Microsoft.Extensions.Logging;

namespace BackPropagation;

public sealed class NeuralNetwork
{
    private ILogger Logger { get; init; }
    private string[] Features;
    private double[][] Data;
    private string OutputFeature;
    private int L { get; init; } //Number of layers
    private int[] N { get; init; } //Units per layer
    private double[][] H { get; init; } // Fields
    private double[] Xi { get; init; } // Activations
    private double[][,] W { get; init; } // Weights
    private double[][] Theta { get; init; } //Thresholds
    private double[][] Delta { get; init; } // Error propagation
    private double[][,] Dw { get; init; } //Weight changes
    private double[][] Dtheta { get; init; } //Threshold changes
    private double[][,] DwPrev { get; init; } //Weight previous changes
    private double[][] DthetaPrev { get; init; } //Threshold previous changes
    private ActivationFunction Fact { get; init; }

    public NeuralNetwork(ILogger logger, string[] features, double[][] data, string outputFeature, int[] unitsPerLayer)
    {
        Features = features;
        Logger = logger;
        L = unitsPerLayer.Length;
        N = unitsPerLayer;
        
        H = new double[L][];
        for(var layer = 0; layer < L ; layer++)
        {
            H[layer] = new double[N[layer]];
        }
    }
    
}