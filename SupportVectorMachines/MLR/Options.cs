using CommandLine;

namespace MLR;

public sealed class Options
{
    
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
    
    [Option('l', "threshold", Required = false, Default = 0.5, HelpText = "Threshold to use when classifying.")]
    public required double Threshold { get; init; }
}