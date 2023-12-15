using CommandLine;

namespace SOM;

public sealed class Options
{
    [Option(shortName: 'i', longName: "input", Required = true, HelpText = "Input file")]
    public required string InputFile { get; init; }

    [Option(shortName: 's', longName: "separator", Required = false, Default = "\t", HelpText = "Separator")]
    public required string Separator { get; init; }
    
    [Option(shortName: 'n', longName: "no-header", Required = false, Default = false, HelpText = "Output file")]
    public required bool NoHeader { get; init; }
    
    [Option(shortName: 'r', longName: "learning-rate", Required = false, Default = 0.001, HelpText = "Learning rate [0, 1]")]
    public required double LearningRate { get; init; }
    
    [Option(shortName: 'w', longName: "width", Required = false, Default = 10, HelpText = "NN width")]
    public required int Width { get; init; }
    
    [Option(shortName: 'h', longName: "height", Required = false, Default = 10, HelpText = "NN height")]
    public required int Height { get; init; }
    
    [Option(shortName: 'e', longName: "epochs", Required = false, Default = 1000, HelpText = "Maximum epochs")]
    public required int Epochs { get; init; }
}