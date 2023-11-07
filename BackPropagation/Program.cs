using BackPropagation.Configuration;
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