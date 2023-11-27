using CommandLine;

namespace DataNormalizer;

public class Options
{
    [Option('i', "input", Required = true, HelpText = "Input file.")]
    public string InputFile { get; set; }

    [Option('d', "delimiter", Default = "\t", HelpText = "Delimiter.")]
    public string Delimiter { get; set; }

    [Option('s', "scaling", Required = true, HelpText = "Scaling methods.")]
    public IEnumerable<string> ScalingMethods { get; set; }

    [Option('m', "minmax", HelpText = "Scale for min-max scaling.")]
    public double MinMaxScale { get; set; }

    [Option('f', "features", HelpText = "Feature names to be scaled.")]
    public IEnumerable<string> FeatureNames { get; set; }
}