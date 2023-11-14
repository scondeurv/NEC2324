using BackPropagation;
using BackPropagation.Configuration;
using BackPropagation.Scaling;
using BackPropagation.Validation;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;

using var loggerFactory = LoggerFactory.Create(builder =>
{
    builder
        .AddFilter("Microsoft", LogLevel.Warning)
        .AddFilter("System", LogLevel.Warning)
        .AddFilter("BackPropagation.Program", LogLevel.Debug)
        .AddConsole();
});

var logger = loggerFactory.CreateLogger<MyNeuralNetwork>();
var parametersFile = args[0];

if (string.IsNullOrWhiteSpace(parametersFile))
{
    logger.LogError("Parameter files is mandatory!");
    return;
}

var json = await File.ReadAllTextAsync(parametersFile);
var parameters = JsonConvert.DeserializeObject<NeuralNetworkParameters>(json);

if (!ValidateNeuralNetworkParameters(parameters, logger))
{
    return;
}

var dataFile = new DataFile();
await dataFile.Load(parameters.DataFile);

var cancellationTokenSource = new CancellationTokenSource();
try
{
    var nn = new MyNeuralNetwork(logger, parameters.UnitsPerLayer,
        parameters.ActivationFunction, parameters.Epochs, parameters.ValidationPercentage, parameters.LearningRate,
        parameters.Momentum);
    
    var factory = new ScalingMethodFactory();
    var scalingPerFeature = factory.CreatePerFeature(parameters.ScalingConfiguration);
    var data = dataFile.Data;
    if (scalingPerFeature.Any())
    {
        logger.LogInformation("Scaling data...");
        var scaler = new DataScaler();
        data = await scaler.Scale(dataFile.Data, dataFile.Features, scalingPerFeature, cancellationTokenSource.Token);
        var scaledDataFile = new DataFile(dataFile.Features, data);
        await scaledDataFile.Save($"{Path.GetFileNameWithoutExtension(parameters.DataFile)}-scaled.txt");
    }
        
    await nn.Fit(data, dataFile.Features, cancellationTokenSource.Token);
    var predictions = await nn.Predict(data, cancellationTokenSource.Token);
    
    var errors = nn.LossEpochs();
    logger.LogInformation("Exporting plots...");
    var plotExporter = new PlotExporter();
    var filename = Path.GetFileNameWithoutExtension(parameters.DataFile);
    plotExporter.ExportLinear(
        $"Error vs Epoch (\u03B7: {parameters.LearningRate:F4}, \u03B1: {parameters.Momentum:F4})",
        "Epoch",
        "Error",
        new Dictionary<string, (double X, double Y)[]>
        {
            { "Training", errors.TrainingErrors.Select((e, epoch) => ((double)epoch, e)).ToArray() },
            { "Validation", errors.ValidationErrors.Select((e, epoch) => ((double)epoch, e)).ToArray() }
        },
        $"{filename}-error-{DateTime.Now:yyyyMMddhhmmss}");
    
    plotExporter.ExportScatter(
        $"Actual vs Prediction (\u03B7: {parameters.LearningRate:F4}, \u03B1: {parameters.Momentum:F4})",
        "Actual",
        "Prediction",
        data.Select((pattern, index) => (pattern[^1], predictions[index])).ToArray(),
        $"{filename}-scatter-{DateTime.Now:yyyyMMddhhmmss}");
    
    logger.LogInformation("Work done");
}
catch
{
    cancellationTokenSource.Cancel();
    throw;
}

static bool ValidateNeuralNetworkParameters(NeuralNetworkParameters trainingParameters, ILogger logger)
{
    var validator = new NeuralNetworksParameterValidator();
    var result = validator.Validate(trainingParameters);

    if (!result.IsValid)
    {
        foreach (var error in result.Errors)
        {
            logger.LogError(error.ErrorMessage);
        }
    }

    return result.IsValid;
}