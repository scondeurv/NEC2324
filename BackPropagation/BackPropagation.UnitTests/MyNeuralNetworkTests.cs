using AutoFixture.Xunit2;
using Microsoft.Extensions.Logging;
using Moq;
using static BackPropagation.ActivationFunctionType;

namespace BackPropagation.UnitTests;

[CollectionDefinition("MyNeuralNetworkTests", DisableParallelization = true)]
public class MyNeuralNetworkTests
{
    private readonly Mock<ILogger> _loggerMock;

    public MyNeuralNetworkTests()
    {
        _loggerMock = new Mock<ILogger>();
    }

    [Theory]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 5, 1 }, new[] { Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 1 }, new[] { Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 13, 9, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 3, 9, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 50, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 300, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.1, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.001, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.4, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { Sigmoid, Sigmoid, Sigmoid, Sigmoid }, 150, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { Linear, Linear, Linear, Linear }, 150, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 4, 9, 5, 1 }, new[] { ReLu, ReLu, ReLu, ReLu }, 150, 1000, 0.2, 0.01, 0.8, "A1-turbine.training.txt", "A1-turbine.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { Linear, Linear, Linear, Linear }, 140, 1000, 0.1, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 7, 1 }, new[] { Linear, Linear, Linear }, 140, 1000, 0.2, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7 }, new[] { Linear, Linear, Linear }, 140, 1000, 0.2, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 23, 15, 7, 1 }, new[] { Linear, Linear, Linear, Linear, Linear }, 140, 10000, 0.2, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 7, 15, 7, 1 }, new[] { Linear, Linear, Linear, Linear, Linear }, 140, 10000, 0.2, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { Linear, Linear, Linear, Linear }, 40, 1000, 0.2, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { Linear, Linear, Linear, Linear }, 400, 1000, 0.2, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { Linear, Linear, Linear, Linear }, 140, 1000, 0.2, 0.1, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { Linear, Linear, Linear, Linear }, 140, 1000, 0.2, 0.00001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { Linear, Linear, Linear, Linear }, 140, 1000, 0.2, 0.0001, 0, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { Linear, Linear, Linear, Linear }, 140, 1000, 0.2, 0.0001, 0.4, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { Sigmoid, Sigmoid, Sigmoid, Sigmoid }, 140, 1000, 0.2, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 140, 1000, 0.2, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    // [InlineAutoData(new[] { 9, 15, 7, 1 }, new[] { ReLu, ReLu, ReLu, ReLu }, 140, 1000, 0.2, 0.0001, 0.6, "A1-synthetic.training.txt", "A1-synthetic.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 5, 1 }, new[] { Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 1 }, new[] { Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 13, 11, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 3, 11, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 50, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 300, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.1, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.001, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { Tanh, Tanh, Tanh, Tanh }, 150, 1000, 0.2, 0.01, 0.4, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { Sigmoid, Sigmoid, Sigmoid, Sigmoid }, 150, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { Linear, Linear, Linear, Linear }, 150, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    [InlineAutoData(new[] { 8, 11, 5, 1 }, new[] { ReLu, ReLu, ReLu, ReLu }, 150, 1000, 0.2, 0.01, 0.8, "market_data.training.txt", "market_data.test.txt")]
    public async Task RunTests(int[] unitsPerLayer, ActivationFunctionType[] facts, int epochs, int? batchSize,
        double validationPercentage, double learningRate, double momentum, string trainingFileName, string testFileName)
    {
        //Arrange
        var trainingFile = new DataFile();
        await trainingFile.Load($"Datasets/{trainingFileName}");

        var testFile = new DataFile();
        await testFile.Load($"Datasets/{testFileName}");

        var nn = new MyNeuralNetwork(_loggerMock.Object, unitsPerLayer, facts, epochs, batchSize, validationPercentage,
            learningRate, momentum);

        //Act
        await nn.Fit(trainingFile.Data, trainingFile.Features);
        var errors = nn.LossEpochs();

        var plotExporter = new PlotExporter();
        var legend =
            $"\u03B7: {learningRate:F4}\n\u03B1: {momentum:F4}\nlayers: {unitsPerLayer.Length}\nepochs: {epochs}\nunits: {string.Join(",", unitsPerLayer)}\nact.: {string.Join(",", facts)}";
        var filename = Path.GetFileNameWithoutExtension($"Datasets/{trainingFileName}");
        plotExporter.ExportLinear(
            $"{filename} - MAPE vs Epoch",
            "Epoch",
            "MAPE",
            new Dictionary<string, (double X, double Y)[]>
            {
                { "Training", errors.TrainingErrors.Select((e, epoch) => ((double)epoch, e.Mape)).ToArray() },
                { "Validation", errors.ValidationErrors.Select((e, epoch) => ((double)epoch, e.Mape)).ToArray() }
            },
            $"{filename}.mape.{DateTime.Now:yyyyMMddhhmmssfff}",
            $"{legend}\nMAPE Training: {errors.TrainingErrors[^1].Mape:F4}\nMAPE Validation: {errors.ValidationErrors[^1].Mape:F4}");


        var testFilename = Path.GetFileNameWithoutExtension($"Datasets/{testFileName}");

        var predictions = await nn.Predict(testFile.Data);

        plotExporter.ExportScatter(
            $"{testFilename} - Actual vs Prediction",
            "Actual",
            "Prediction",
            testFile.Data.Select((pattern, index) => (pattern[^1], predictions[index])).ToArray(),
            $"{filename}.scatter.{DateTime.Now:yyyyMMddhhmmssfff}",
            legend);
    }
}