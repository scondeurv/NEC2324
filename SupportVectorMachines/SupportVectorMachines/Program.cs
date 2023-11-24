using System.Globalization;
using LibSVMsharp;
using LibSVMsharp.Helpers;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Series;

var problem = new SVMProblem();
var problemDataset = File.ReadLinesAsync(@"Datasets/A2-ring-merged.txt");

await foreach (var line in problemDataset)
{
    var split = line.Split('\t');
    var y = double.Parse(split[^1], CultureInfo.InvariantCulture);
    var x = new SVMNode[split.Length - 1];
    for (var i = 0; i < split.Length - 1; i++)
    {
        x[i] = new SVMNode(i + 1, double.Parse(split[i], CultureInfo.InvariantCulture));
    }

    problem.Add(x, y);
}

var testProblem = new SVMProblem();
var testDataset = File.ReadLinesAsync(@"Datasets/A2-ring-test.txt");
await foreach (var line in testDataset)
{
    var split = line.Split('\t');
    var y = double.Parse(split[^1], CultureInfo.InvariantCulture);
    var x = new SVMNode[split.Length - 1];
    for (var i = 0; i < split.Length - 1; i++)
    {
        x[i] = new SVMNode(i + 1, double.Parse(split[i], CultureInfo.InvariantCulture));
    }

    testProblem.Add(x, y);
}

// var parameter = new SVMParameter
// {
//     Type = SVMType.NU_SVC,
//     Kernel = SVMKernelType.RBF,
//     Nu = 0.01,
// };

var parameter = new SVMParameter
{
    Type = SVMType.NU_SVC,
    Kernel = SVMKernelType.RBF,
    Nu = 0.55,
    Gamma = 4000
};

var model = SVM.Train(problem, parameter);

var target = new double[testProblem.Length];
for (var i = 0; i < testProblem.Length; i++)
{
    target[i] = SVM.Predict(model, testProblem.X[i]);
}

var accuracy = SVMHelper.EvaluateClassificationProblem(testProblem, target);
Console.WriteLine($"Accuracy: {accuracy}");

PlotModel CreatePlotModel(string title, ScatterSeries class1Series, ScatterSeries class2Series)
{
    var plotModel = new PlotModel { Title = title };
    plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Feature 1" });
    plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Feature 2" });
    plotModel.Series.Add(class1Series);
    plotModel.Series.Add(class2Series);
    return plotModel;
}

// Create series for actual values
var actualClass1Series = new ScatterSeries
    { Title = "Actual Class 1", MarkerType = MarkerType.Circle, MarkerFill = OxyColors.Green };
var actualClass2Series = new ScatterSeries
    { Title = "Actual Class 2", MarkerType = MarkerType.Circle, MarkerFill = OxyColors.Orange };

// Add actual data points to their respective series
foreach (var vec in problem.X.Zip(problem.Y, (x, y) => new { X = x, Y = y }))
{
    var point = new ScatterPoint(vec.X[0].Value, vec.X[1].Value);
    if (vec.Y == 1)
        actualClass1Series.Points.Add(point);
    else
        actualClass2Series.Points.Add(point);
}

// Create series for predicted values
var predictedClass1Series = new ScatterSeries
    { Title = "Predicted Class 1", MarkerType = MarkerType.Diamond, MarkerFill = OxyColors.Green };
var predictedClass2Series = new ScatterSeries
    { Title = "Predicted Class 2", MarkerType = MarkerType.Diamond, MarkerFill = OxyColors.Orange };

// Add predicted data points to their respective series
for (var i = 0; i < testProblem.Length; i++)
{
    var point = new ScatterPoint(testProblem.X[i][0].Value, testProblem.X[i][1].Value);
    if (target[i] == 1)
        predictedClass1Series.Points.Add(point);
    else
        predictedClass2Series.Points.Add(point);
}

// Create plot models
var actualPlotModel = CreatePlotModel("Actual SVM Classification Results", actualClass1Series, actualClass2Series);
var predictedPlotModel =
    CreatePlotModel("Predicted SVM Classification Results", predictedClass1Series, predictedClass2Series);

actualPlotModel.Background = OxyColors.White;
predictedPlotModel.Background = OxyColors.White;

// Export the actual values plot to a PNG image
var pngExporter = new PngExporter { Width = 600, Height = 400 };
var actualPath = "svm_actual_classification_results.png";
using (var stream = File.Create(actualPath))
{
    pngExporter.Export(actualPlotModel, stream);
}

Console.WriteLine($"Actual values plot has been saved to '{actualPath}'");

// Export the predicted values plot to a PNG image
var predictedPath = "svm_predicted_classification_results.png";
using (var stream = File.Create(predictedPath))
{
    pngExporter.Export(predictedPlotModel, stream);
}

Console.WriteLine($"Predicted values plot has been saved to '{predictedPath}'");