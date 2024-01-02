using CommandLine;

namespace TSP;

public sealed class Options
{
    [Option(shortName: 'p', longName: "problem", HelpText = "Problem name", Group = "Exec mode")]
    public required string Problem { get; init; }

    [Option(shortName: 'l', longName: "list", HelpText = "List available problems", Group = "Exec mode")]
    public required bool ListProblems { get; init; }

    [Option(shortName: 'i', longName: "max-iterations", Required = false, Default = 10,
        HelpText = "Max iterations. Iterations get divided by max threads")]
    public required int MaxIterations { get; init; }

    [Option(shortName: 't', longName: "threads", Required = false, Default = 1,
        HelpText = "Max parallel threads. Iterations get divided by this number")]
    public required int MaxThreads { get; init; }

    [Option(shortName: 'g', longName: "generations", Required = false, Default = 25, HelpText = "Max generations")]
    public required int MaxGenerations { get; init; }

    [Option(shortName: 'j', longName: "population-adjust", Required = false, Default = 1.5,
        HelpText = "Population adjust factor. It is multiplied by the genes amount to get the population size")]
    public required double PopulationAdjustFactor { get; init; }

    [Option(shortName: 'm', longName: "population-adjust-multiplier", Required = false, Default = 1.25,
        HelpText =
            "Population adjust factor multiplier. It is multiplied by populations adjust factor every iteration")]
    public required double PopulationAdjustMultiplier { get; init; }
    
    [Option(shortName: 's', longName: "selection", Required = false, Default = "rank",
        HelpText =
            "Selection method. Possible values: rank, tournament")]
    public required string SelectionMethod { get; init; }
    
    [Option(shortName: 'c', longName: "crossover", Required = false, Default = "ox",
        HelpText =
            "Crossover method. Possible values: ox, pmx")]
    public required string CrossoverMethod { get; init; }
    
    [Option(shortName: 'x', longName: "mutation", Required = false, Default = "inversion",
        HelpText =
            "Mutation method. Possible values: inversion, right-shift")]
    public required string MutationMethod { get; init; }
    
    [Option(shortName: 'e', longName: "elitism", Required = false, Default = 0.1,
        HelpText =
            "Elitism. Percentage of best chromosomes to keep from one generation to the next")]
    public required double ElitesPercentage { get; init; }
    
    [Option(shortName: 'h', longName: "stall-threshold", Required = false, Default = 0.1,
        HelpText =
            "Stall threshold. Percentage of generations without improvement to consider the algorithm stalled")]
    public required double StallThreshold { get; init; }

    [Option(shortName: 'd', longName: "distance", Required = false, Default = "", Group = "Exec mode",
        HelpText =
            "Get distance of a tour defined in a file with the nodes separated by commas")]
    public string Distance { get; init; }
}