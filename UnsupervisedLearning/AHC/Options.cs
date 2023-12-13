using CommandLine;

namespace AHC;

public sealed class Options
{
    [Option(shortName: 'i', longName: "input", Required = true, HelpText = "Input file")]
    public required string InputFile { get; init; }

    [Option(shortName: 's', longName: "separator", Required = false, Default = "\t", HelpText = "Separator")]
    public required string Separator { get; init; }
    
    [Option(shortName: 'h', longName: "no-header", Required = false, Default = false, HelpText = "Output file")]
    public required bool NoHeader { get; init; }
    
    [Option(shortName: 'l', longName: "linkage", Required = false, Default = "upgma", HelpText = "Linkage: complete, upgma")]
    public required string Linkage { get; init; }
}