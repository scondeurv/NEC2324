using CommandLine;

namespace KMeans;

public sealed class Options
{
    [Option(shortName: 'i', longName: "input", Required = true, HelpText = "Input file")]
    public required string InputFile { get; init; }

    [Option(shortName: 'd', longName: "delimiter", Required = false, Default = "\t", HelpText = "Delimiter")]
    public required string Delimiter { get; init; }
    
    [Option(shortName: 'h', longName: "no-header", Required = false, Default = false, HelpText = "Output file")]
    public required bool NoHeader { get; init; }
    
    [Option(shortName: 'k', longName: "kvalue", Required = false, Default = 2, HelpText = "KMeans K")]
    public required int K { get; init; }
}