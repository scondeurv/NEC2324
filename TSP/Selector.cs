namespace TSP;

public static class Selector
{
    private static Random Random { get; } = new();
    
    public static Chromosome DoSelection(string method, IEnumerable<Chromosome> population, 
        IReadOnlyDictionary<string, object>? methodParams = null)
    {
        switch (method.ToLower())
        {
            case "rank":
                return RankSelection(population);
            case "tournament":
                return TournamentSelection(population, methodParams is not null ? (int) methodParams["TournamentSize"] : 3);
            default:
                throw new NotSupportedException(method);
        }
    }

    public static Chromosome RankSelection(IEnumerable<Chromosome> population)
    {
        var sortedPopulation = population.OrderByDescending(c => c.Fitness).ToArray();
        var rankSum = sortedPopulation.Length;
        var rankPosition = Random.Next(rankSum);
        return sortedPopulation[rankPosition];
    }

    public static Chromosome TournamentSelection(IEnumerable<Chromosome> population, int tournamentSize)
    {
        var tournament = new Chromosome[tournamentSize];
        var populationArray = population.ToArray();
        for (var i = 0; i < tournamentSize; i++)
        {
            tournament[i] = populationArray[Random.Next(populationArray.Length)];
        }

        return tournament.OrderByDescending(c => c.Fitness).First();
    }
}