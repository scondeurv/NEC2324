using CommandLine;

namespace SupportVectorMachines.RunSVM;

public sealed class Options
{
    [Option('m', "model", Required = false, HelpText = "Model file to read.")]
    public required string ModelFile { get; init; }
    
    [Option('x', "export-plots", Required = false, Default = false, HelpText = "Export plots to compare actual values with predicted values.")]
    public required bool ExportPlots { get; init; }
    
    [Option('u', "plot-features", Required = false, Default = null, HelpText = "Features to plot (f1:f2).")]
    public required string? PlotFeatures  { get; init; }
    
    [Option('t', "training-file", Required = true, HelpText = "Training file to read.")]
    public required string DatasetFile { get; init; }
    
    [Option('e', "test-file", Required = false, HelpText = "Test dataset file to read.")]
    public required string TestFile { get; init; }
    
    [Option('d', "delimiter", Required = false, Default = "\t", HelpText = "Delimiter to use when reading the input file.")]
    public required string Delimiter { get; init; }
    
    [Option('h', "no-header", Required = false, Default = false, HelpText = "Delimiter to use when reading the input file.")]
    public required bool NoHeader { get; init; }
    
    [Option('p', "training-percentage", Required = false, Default = 0.8, HelpText = "Percentage of the dataset to use for training.")]
    public required double TrainingPercentage { get; init; }
    
    [Option('o', "optimizer", Required = false, Default = "random", HelpText = "Optimizer to use (random or search).")]
    public required string Optimizer { get; init; }
    
    [Option('i', "iterations", Required = false, Default = 100, HelpText = "Iterations to use when using the random optimizer.")]
    public required int Iterations { get; init; }
    
    [Option('f', "fscore", Required = false, Default = 0.8, HelpText = "Fscore target")]
    public required double FScore { get; init; }
    
    [Option('s', "svc", Required = false, Default = "c", HelpText = "SVC type (c or nu).")]
    public required string Svc { get; init; }
    
    [Option('k', "kernel", Required = false, Default = "poly", HelpText = "Kernel type (linear, poly, rbf, sigmoid).")]
    public required string Kernel { get; init; }
}