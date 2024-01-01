using System.Collections.Immutable;
using CommandLine;
using TSP;
using TspLibNet;

Parser.Default.ParseArguments<Options>(args).WithParsedAsync(async opts =>
{
    #region List problems

    if (opts.ListProblems)
    {
        var sep = Path.DirectorySeparatorChar;
        Console.WriteLine("Available TSP problems:");
        Directory
            .EnumerateFiles($".{sep}TSPLIB95{sep}TSP")
            .ToList()
            .ForEach(f => Console.WriteLine(Path.GetFileNameWithoutExtension(f)));
        return;
    }

    #endregion


    #region Run TSP problem

    if (opts.Problem != null)
    {
        var problem = LoadProblem(opts.Problem);
        var populationAdjustFactor = opts.PopulationAdjustFactor;
        var generations = opts.MaxGenerations;
        var configurations = new List<Configuration>();
        Chromosome bestChromosome = default;
        var taskFactory = new TaskFactory<Chromosome>();
        Console.WriteLine("Fitting population size...");
        for (var i = 0; i < opts.MaxIterations / opts.MaxThreads; i++)
        {
            Console.WriteLine($"Iteration {i}");
            var tasks = new List<Task<Chromosome>>();
            for (var j = 0; j < opts.MaxThreads; j++)
            {
                var population = InitPopulation(populationAdjustFactor, problem);
                var task = taskFactory.StartNew(
                    () => Run(population, generations, problem, opts.SelectionMethod, opts.CrossoverMethod,
                        opts.MutationMethod, id: j, elitesPercentage: opts.ElitesPercentage, stallThreshold: opts.StallThreshold),
                    TaskCreationOptions.LongRunning | TaskCreationOptions.AttachedToParent);
                tasks.Add(task);
                populationAdjustFactor *= opts.PopulationAdjustMultiplier;
            }

            Task.WaitAll(tasks.ToArray());
            var best = tasks.Select(t => t.GetAwaiter().GetResult())
                .Select((c, index) => (c, index))
                .MaxBy(t => t.c.Fitness);

            var populationSizeMultiplier = populationAdjustFactor / Math.Pow(opts.PopulationAdjustMultiplier, (opts.MaxThreads - best.index));

            if (bestChromosome is null || best.c.Fitness > bestChromosome.Fitness)
            {
                bestChromosome = best.c;
            }

            Console.WriteLine($"Fitting population size. Current best fitness: {bestChromosome.Fitness}");
            configurations.Add(new Configuration(populationSizeMultiplier, generations, best.c.Fitness));
        }

        var bestConfiguration = configurations.OrderByDescending(c => c.Fitness).First();
        
        Console.WriteLine("---------------------");
        Console.WriteLine("BEST SOLUTION FOUND:");
        Console.WriteLine("---------------------");
        Console.WriteLine(bestConfiguration);
        Console.WriteLine(bestChromosome);
        Console.WriteLine($"Distance: {1 / bestChromosome.Fitness}");
    }

    #endregion
});

static IProblem LoadProblem(string problemName)
{
    var sep = Path.DirectorySeparatorChar;
    var tspLib = new TspLib95(Path.GetFullPath($".{sep}TSPLIB95"));
    tspLib.LoadTSP(problemName);
    var tsp = tspLib.TSPItems().First();
    Console.WriteLine(tsp.ToString());
    return tsp.Problem;
}

static Chromosome Run(
    ImmutableArray<Chromosome> sourcePopulation,
    int generations,
    IProblem problem,
    string selectionMethod,
    string crossoverMethod,
    string mutationMethod,
    double elitesPercentage = 0.0,
    double stallThreshold = 0.25,
    IReadOnlyDictionary<string, object> methodParams = null,
    int id = 0)
{
    Console.WriteLine(
        $"[T-{id}] Evolving a population of {sourcePopulation.Length} along {generations} generations...");
    var population = sourcePopulation.ToArray();
    double previousBestFitness = 0;
    var stallCounter = 0;
    for (var generation = 0; generation < generations; generation++)
    {
        var populationPrime = new List<Chromosome>();
        for (var pair = 0; pair < population.Length / 2; pair++)
        {
            var c1 = Selector.DoSelection(selectionMethod, population, methodParams);
            var c2 = Selector.DoSelection(selectionMethod, population, methodParams);
            var c1Prime = Crossover.DoCrossover(crossoverMethod, c1, c2, problem);
            var c2Prime = Crossover.DoCrossover(crossoverMethod, c2, c1, problem);
            populationPrime.Add(Mutator.DoMutation(mutationMethod, c1Prime, problem));
            populationPrime.Add(Mutator.DoMutation(mutationMethod, c2Prime, problem));
        }

        if (elitesPercentage > 0)
        {
            populationPrime.AddRange(ApplyElitism(population, elitesPercentage));
        }

        population = populationPrime.ToArray();
        var best = population.OrderByDescending(c => c.Fitness).First();
        
        if (best.Fitness <= previousBestFitness)
        {
            stallCounter++;
            if (stallCounter >= generations * (1 + stallThreshold))
            {
                Console.WriteLine($"[T-{id}] GA is stalled.");
                break;
            }
        }
        else
        {
            stallCounter = 0;
            previousBestFitness = best.Fitness;
        }
    }

    var winner = population.OrderByDescending(c => c.Fitness).First();
    Console.WriteLine($"[T-{id}] Winner: {winner}");
    return winner;
}

static ImmutableArray<Chromosome> InitPopulation(double geneMultiplier, IProblem problem)
{
    var population = new List<Chromosome>();
    var populationSize = (int)(geneMultiplier * problem.NodeProvider.GetNodes().Count);
    for (var i = 0; i < populationSize; i++)
    {
        var chromosome = ChromosomeFactory.Create(problem);
        population.Add(chromosome);
    }

    return population.ToImmutableArray();
}

static IEnumerable<Chromosome> ApplyElitism(IEnumerable<Chromosome> population, double elitismPercentage)
{
    var elitesCount = (int)(population.Count() * elitismPercentage);
    var sortedPopulation = population.OrderByDescending(c => c.Fitness).ToList();
    var elites = sortedPopulation.Take(elitesCount).ToList();
    return elites;
}