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

var trainingFile = new DataFile();
await trainingFile.Load(parameters.TrainingFile);

DataFile testFile = null;
if (!string.IsNullOrWhiteSpace(parameters.TestFile))
{
    testFile = new DataFile();
    await testFile.Load(parameters.TestFile);
}

var cancellationTokenSource = new CancellationTokenSource();
try
{
    var nn = new MyNeuralNetwork(logger, parameters.UnitsPerLayer,
        parameters.ActivationFunctionPerLayer, parameters.Epochs, parameters.ValidationPercentage, parameters.LearningRate,
        parameters.Momentum);
    
    var factory = new ScalingMethodFactory();
    var scalingPerFeature = factory.CreatePerFeature(parameters.ScalingConfiguration);
    var trainingData = trainingFile.Data;
    var testData = testFile?.Data;
    if (scalingPerFeature.Any())
    {
        logger.LogInformation("Scaling data...");
        var scaler = new DataScaler();
        trainingData = await scaler.Scale(trainingFile.Data, trainingFile.Features, scalingPerFeature, cancellationTokenSource.Token);
        var scaledTrainingDataFile = new DataFile(trainingFile.Features, trainingData);
        await scaledTrainingDataFile.Save($"{Path.GetFileNameWithoutExtension(parameters.TrainingFile)}.scaled.txt");

        if (testData != null)
        {
            testData = await scaler.Scale(testFile.Data, testFile.Features, scalingPerFeature,
                cancellationTokenSource.Token);
            var scaledTestDataFile = new DataFile(testFile.Features, testData);
            await scaledTestDataFile.Save($"{Path.GetFileNameWithoutExtension(parameters.TestFile)}.scaled.txt");
        }
    }
        
    await nn.Fit(trainingData, trainingFile.Features, cancellationTokenSource.Token);
    
    var errors = nn.LossEpochs();
    logger.LogInformation($"Training MAPE: {errors.TrainingErrors[^1].Mape}" );
    logger.LogInformation($"Validation MAPE: {errors.ValidationErrors[^1].Mape}" );
    var plotExporter = new PlotExporter();
    var legend =
        $"\u03B7: {parameters.LearningRate:F4}\n\u03B1: {parameters.Momentum:F4}\nlayers: {parameters.Layers}\nepochs: {parameters.Epochs}\nunits: {string.Join(",", parameters.UnitsPerLayer)}\nact.: {string.Join(",", parameters.ActivationFunctionPerLayer)}";
    var filename = Path.GetFileNameWithoutExtension(parameters.TrainingFile);
    plotExporter.ExportLinear(
        $"{filename} - MAPE vs Epoch",
        "Epoch",
        "MAPE",
        new Dictionary<string, (double X, double Y)[]>
        {
            { "Training", errors.TrainingErrors.Select((e, epoch) => ((double)epoch, e.Mape)).ToArray() },
            { "Validation", errors.ValidationErrors.Select((e, epoch) => ((double)epoch, e.Mape)).ToArray() }
        },
        $"{filename}.mape.{DateTime.Now:yyyyMMddhhmmss}",
        $"{legend}\nMAPE Training: {errors.TrainingErrors[^1].Mape:F4}\nMAPE Validation: {errors.ValidationErrors[^1].Mape:F4}");
    
    plotExporter.ExportLinear(
        $"{filename} - MSE vs Epoch",
        "Epoch",
        "MSE",
        new Dictionary<string, (double X, double Y)[]>
        {
            { "Training", errors.TrainingErrors.Select((e, epoch) => ((double)epoch, e.Mse)).ToArray() },
            { "Validation", errors.ValidationErrors.Select((e, epoch) => ((double)epoch, e.Mse)).ToArray() }
        },
        $"{filename}.mse.{DateTime.Now:yyyyMMddhhmmss}",
        legend);

    if (testData != null)
    {
        var testFilename = Path.GetFileNameWithoutExtension(parameters.TestFile);
        logger.LogInformation("Testing ...");
        var predictions = await nn.Predict(testData, cancellationTokenSource.Token);

        plotExporter.ExportScatter(
            $"{testFilename} - Actual vs Prediction",
            "Actual",
            "Prediction",
            testData.Select((pattern, index) => (pattern[^1], predictions[index])).ToArray(),
            $"{filename}.scatter.{DateTime.Now:yyyyMMddhhmmss}",
            legend);
    }

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