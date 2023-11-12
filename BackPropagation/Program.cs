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

var logger = loggerFactory.CreateLogger<Program>();
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
    var nn = new MyNeuralNetwork(logger, dataFile.Features, dataFile.Data, parameters.UnitsPerLayer,
        parameters.ActivationFunction, parameters.Epochs, parameters.ValidationPercentage);
    var factory = new ScalingMethodFactory();
    var scalingPerFeature = factory.CreatePerFeature(parameters.ScalingConfiguration);
    await nn.Fit(parameters.LearningRate, parameters.Momentum, scalingPerFeature, cancellationTokenSource.Token);
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