using CommandLine;

namespace Tools.Outliers;

public class Options
{
    [Option('f', "input-file", Required = true, HelpText = "Input file.")]
    public required string InputFile { get; init; }

    [Option('d', "delimiter", Default = "\t", HelpText = "Delimiter.")]
    public required string Delimiter { get; init; }

    [Option('h', "no-header", Required = false, Default = false, HelpText = "Delimiter to use when reading the input file.")]
    public required bool NoHeader { get; init; }
    
    [Option('m', "method", Required = false, Default = "iqr", HelpText = "Delimiter to use when reading the input file.")]
    public required string Method { get; init; }
    
    [Option('t', "threshold", Required = false, Default = 3.0, HelpText = "Zscore threshold.")]
    public required double Threshold { get; init; }
    [Option('c', "clean", Required = false, Default = false, HelpText = "Clean the dataset")]
    public required bool Clean { get; init; }
}