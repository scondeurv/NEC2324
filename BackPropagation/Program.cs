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
var trainingParameters = JsonConvert.DeserializeObject<TrainingParameters>(json);

if (!ValidateTrainingParameters(trainingParameters, logger))
{
    return;
}

var dataFile = new DataFile();
await dataFile.Load(trainingParameters.DataFile);

var cancellationTokenSource = new CancellationTokenSource();
try
{
    var nn = new NeuralNetwork(logger, dataFile.Features, dataFile.Data, trainingParameters.UnitsPerLayer,
        trainingParameters.ActivationFunction);
    var factory = new ScalingMethodFactory();
    var scalingPerFeature = factory.CreatePerFeature(trainingParameters.ScalingConfiguration);
    await nn.Train(trainingParameters.Epochs, trainingParameters.TrainingDataPercentage, trainingParameters.TestDataPercentage,
        trainingParameters.LearningRate, trainingParameters.Momentum, scalingPerFeature);
}
catch
{
    cancellationTokenSource.Cancel();
    throw;
}

static bool ValidateTrainingParameters(TrainingParameters trainingParameters, ILogger logger)
{
    var validator = new TrainingParametersValidator();
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