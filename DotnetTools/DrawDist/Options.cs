using CommandLine;

namespace Tools.DrawDist;

public sealed class Options
{
    [Option('f', "input-file", Required = true, HelpText = "Dataset file to read.")]
    public required string InputFile { get; init; }
    
    [Option('d', "delimiter", Required = false, Default = "\t", HelpText = "Delimiter to use when reading the input file.")]
    public required string Delimiter { get; init; }
    
    [Option('h', "no-header", Required = false, Default = false, HelpText = "Delimiter to use when reading the input file.")]
    public required bool NoHeader { get; init; }
}