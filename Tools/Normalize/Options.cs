using CommandLine;

namespace Tools.Normalize;

public class Options
{
    [Option('f', "input-file", Required = true, HelpText = "Input file.")]
    public required string InputFile { get; init; }

    [Option('d', "delimiter", Default = "\t", HelpText = "Delimiter.")]
    public required string Delimiter { get; init; }

    [Option('s', "scaling-per-feature", Required = true, HelpText = "Scaling method to apply per feature (feature:method, feature:method, ...).")]
    public IEnumerable<string> ScalingPerFeature { get; init; }
    
    [Option('h', "no-header", Required = false, Default = false, HelpText = "Delimiter to use when reading the input file.")]
    public required bool NoHeader { get; init; }
}