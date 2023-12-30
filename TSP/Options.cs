using CommandLine;

namespace TSP;

public sealed class Options
{
    [Option(shortName: 'p', longName: "problem", HelpText = "Problem name", Group = "Exec mode")]
    public required string Problem { get; init; }

    [Option(shortName: 'l', longName: "list", HelpText = "List available problems", Group = "Exec mode")]
    public required bool ListProblems { get; init; }

    [Option(shortName: 'i', longName: "max-iterations", Required = false, Default = 100,
        HelpText = "[Fit mode] Max iterations. Iterations get divided by max threads. Run twice, one to fit the population size and then to fit the generations")]
    public required int MaxIterations { get; init; }

    [Option(shortName: 't', longName: "threads", Required = false, Default = 5,
        HelpText = "[Fit mode] Max parallel threads. Iterations get divided by this number")]
    public required int MaxThreads { get; init; }

    [Option(shortName: 'g', longName: "generations", Required = false, Default = 100, HelpText = "Max generations")]
    public required int MaxGenerations { get; init; }

    [Option(shortName: 'a', longName: "generations-adjust", Required = false, Default = -10,
        HelpText = "[Fit mode] Generations adjust factor. It is added to the max generations every iteration")]
    public required int GenerationAdjustFactor { get; init; }

    [Option(shortName: 'j', longName: "population-adjust", Required = false, Default = 1.5,
        HelpText = "Population adjust factor. It is multiplied by the genes amount to get the population size")]
    public required double PopulationAdjustFactor { get; init; }

    [Option(shortName: 'm', longName: "population-adjust-multiplier", Required = false, Default = 1.1,
        HelpText =
            "[Fit mode] Population adjust factor multiplier. It is multiplied by populations adjust factor every iteration")]
    public required double PopulationAdjustMultiplier { get; init; }
    
    [Option(shortName: 'r', longName: "improvement-threshold", Required = false, Default = .01,
        HelpText =
            "[Fit mode] Minimum improvement threshold. If the best fitness is not improved by this factor, the algorithm stops")]
    public required double MinimumImprovementThreshold { get; init; }
}