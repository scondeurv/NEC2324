namespace TSP;

public static class Selection
{
    public static Chromosome RankSelection(IEnumerable<Chromosome>  population, Random random)
    {
        var sortedPopulation = population.OrderBy(c => c.Fitness).ToArray();
        var rankSum = sortedPopulation.Select((c, i) => i + 1).Sum();
        var rankPosition = random.Next(rankSum);
        var rankCounter = 0;

        for (var i = 0; i < sortedPopulation.Length; i++)
        {
            rankCounter += i + 1;
            if (rankCounter >= rankPosition)
            {
               return sortedPopulation[i];
            }
        }

        return null!;
    }

    public static Chromosome TournamentSelection(IEnumerable<Chromosome> population,  Random random, int tournamentSize)
    {
        var tournament = new Chromosome[tournamentSize];
        var populationArray = population.ToArray();
        for (var i = 0; i < tournamentSize; i++)
        {
            tournament[i] = populationArray[random.Next(populationArray.Length)];
        }

        return tournament.OrderByDescending(c => c.Fitness).First();
    }
}