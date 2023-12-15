using Accord.Neuro;
using Accord.Neuro.Learning;
using CommandLine;
using SOM;
using Tools.Common;
using PlotExporter = SOM.PlotExporter;

await Parser.Default.ParseArguments<Options>(args).WithParsedAsync(async opt =>
{
    var dataset = new Dataset();
    await dataset.Load(opt.InputFile, opt.Separator, opt.NoHeader);

    var numInputs = dataset.Data.Count - 1;
    var network = new DistanceNetwork(numInputs, opt.Width * opt.Height );
    var som = new SOMLearning(network, opt.Width, opt.Height)
    {
        LearningRate = opt.LearningRate,
        LearningRadius = (opt.Width > opt.Height ? opt.Width : opt.Height) / 2,
    };

    var (input, _) = dataset.Split();
    
    for (var i = 0; i < opt.Epochs; i++)
    {
        som.RunEpoch(input);
        som.LearningRate = som.LearningRate * 0.99;
        som.LearningRadius = som.LearningRadius * 0.99;
    }

    var clusters = new Dictionary<int, List<int>>();
    for (var pattern = 0; pattern < input.Length; pattern++)
    {
        network.Compute(input[pattern]);
        var winner = network.GetWinner();
        if (!clusters.ContainsKey(winner))
        {
            clusters[winner] = new List<int>();
        }
        clusters[winner].Add(pattern);
    }

    clusters
        .Select(kvp => (kvp.Key, kvp.Value.Count))
        .OrderByDescending(kpv => kpv.Count)
        .ToList()
        .ForEach(kpv => Console.WriteLine($"{kpv.Key}: {kpv.Count}"));

    
    var plotter = new PlotExporter();
    var fileName = $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-heatmap.png";
    var title = $"SOM - {Path.GetFileNameWithoutExtension(opt.InputFile)}";
    plotter.PlotHeatmap(clusters, fileName, opt.Width, opt.Height, title);
    
    title = $"U-Matrix - {Path.GetFileNameWithoutExtension(opt.InputFile)}";
    var umatrix = CalculateUMatrix(network, opt.Width, opt.Height);
    plotter.PlotUMatrix(umatrix, $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-umatrix.png", opt.Width, opt.Height, title);
    
    var features = dataset.Data.Keys.ToArray()[0..^1];
    PlotComponentPlanes(network, opt.Width, opt.Height, plotter, $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-component", features);
});

static void PlotComponentPlanes(DistanceNetwork network, int width, int height, PlotExporter plotExporter, string outputPath, string[] components)
{
    for (var index = 0; index < components.Length; index++)
    {
        var componentPlane = new double[width, height];

        for (var i = 0; i < width; i++)
        {
            for (var j = 0; j < height; j++)
            {
                var neuronWeights = network.Layers[0].Neurons[i * width + j].Weights;
                componentPlane[i, j] = neuronWeights[index];
            }
        }

        var fileName = $"{outputPath}-{components[index]}.png";
        var title = $"Component {components[index]}";
        plotExporter.PlotHeatmap(componentPlane, fileName, width, height, title);
    }
}

static double[,] CalculateUMatrix(DistanceNetwork network, int width, int height)
{
    var uMatrix = new double[width, height];

    for (var i = 0; i < width; i++)
    {
        for (var j = 0; j < height; j++)
        {
            var neuronWeights = network.Layers[0].Neurons[i * width + j].Weights;

            double sum = 0;
            var count = 0;
            
            for (var dx = -1; dx <= 1; dx++)
            {
                for (var dy = -1; dy <= 1; dy++)
                {
                    var nx = i + dx;
                    var ny = j + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                    {
                        var neighborWeights = network.Layers[0].Neurons[nx * width + ny].Weights;
                        sum += CalculateDistance(neuronWeights, neighborWeights);
                        count++;
                    }
                }
            }

            uMatrix[i, j] = sum / count;
        }
    }

    return uMatrix;
}

static double CalculateDistance(double[] weights1, double[] weights2)
{
    double sum = 0;

    for (var i = 0; i < weights1.Length; i++)
    {
        var diff = weights1[i] - weights2[i];
        sum += diff * diff;
    }

    return Math.Sqrt(sum);
}