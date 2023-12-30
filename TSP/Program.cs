using System.Collections.Immutable;
using CommandLine;
using TSP;
using TspLibNet;

Parser.Default.ParseArguments<Options>(args).WithParsedAsync(async opts =>
{
    #region List problems 
    if(opts.ListProblems)
    {
        var sep = Path.DirectorySeparatorChar ;
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
        Chromosome chromosome = default;
        var taskFactory = new TaskFactory<Chromosome>();
        Console.WriteLine("Fitting population size...");
        for (var i = 0; i < opts.MaxIterations / opts.MaxThreads; i++)
        {
            var tasks = new List<Task<Chromosome>>();
            for (var j = 0; j < opts.MaxThreads; j++)
            {
                var population = InitPopulation(populationAdjustFactor, problem);
                var task = taskFactory.StartNew(() => Run(population, generations, problem, j), TaskCreationOptions.LongRunning | TaskCreationOptions.AttachedToParent);
                tasks.Add(task);
                populationAdjustFactor *= opts.PopulationAdjustMultiplier;
            }

            Task.WaitAll(tasks.ToArray());
            var best = tasks.Select(t => t.GetAwaiter().GetResult())
                .Select((c, index) => (c, index))
                .MinBy(t => t.c.Fitness);
            var bestPopulationSizeMultiplier = populationAdjustFactor / Math.Pow(1.01, (opts.MaxThreads - best.index));

            if (configurations.Any(c => c.Fitness * (1 + opts.MinimumImprovementThreshold) < best.c.Fitness))
            {
                break;
            }

            chromosome = best.c;
            Console.WriteLine($"Fitting population size. Current best fitness: {chromosome.Fitness}");
            configurations.Add(new Configuration(bestPopulationSizeMultiplier, generations, chromosome.Fitness));
        }

        var bestConfiguration = configurations.OrderBy(c => c.Fitness).First();
        configurations.Clear();
        configurations.Add(bestConfiguration);
        
        Console.WriteLine("Fitting generations...");
        for (var i = 0; i < opts.MaxIterations / opts.MaxThreads; i++)
        {
            var tasks = new List<Task<Chromosome>>();
            for (var j = 0; j < opts.MaxThreads; j++)
            {
                var population = InitPopulation(bestConfiguration.PopulationSizeMultiplier, problem);
                var task = taskFactory.StartNew(() => Run(population, generations, problem, j), TaskCreationOptions.LongRunning | TaskCreationOptions.AttachedToParent);
                tasks.Add(task);
                generations += opts.GenerationAdjustFactor;
            }

            Task.WaitAll(tasks.ToArray());
            var best = tasks.Select(t => t.GetAwaiter().GetResult())
                .Select((c, index) => (c, index))
                .MinBy(t => t.c.Fitness);
            var bestGenerations = generations - (opts.GenerationAdjustFactor * (opts.MaxThreads - best.index));

            if (generations <= 0 || configurations.Any(c => c.Fitness * (1 + opts.MinimumImprovementThreshold) < best.c.Fitness))
            {
                break;
            }

            chromosome = best.c;
            Console.WriteLine($"Fitting generations. Current best fitness: {best.c.Fitness}");
            configurations.Add(new Configuration(bestConfiguration.PopulationSizeMultiplier, (int)bestGenerations,
                best.c.Fitness));
        }

        bestConfiguration = configurations.OrderBy(c => c.Fitness).First();
        Console.WriteLine(bestConfiguration);
        Console.WriteLine(chromosome);
    }

    #endregion
});

static IProblem LoadProblem(string problemName)
{
    var sep = Path.DirectorySeparatorChar ;
    var tspLib = new TspLib95(Path.GetFullPath($".{sep}TSPLIB95"));
    tspLib.LoadTSP(problemName);
    var tsp = tspLib.TSPItems().First();
    Console.WriteLine(tsp.ToString());
    return tsp.Problem;
}

static Chromosome Run(ImmutableArray<Chromosome> population, int generations, IProblem problem, int id = 0)
{
    Console.WriteLine($"[T-{id}] Evolving a population of {population.Length} along {generations} generations...");
    for (var generation = 0; generation < generations; generation++)
    {
        var populationPrime = new List<Chromosome>();
        var random = new Random();
        for (var pair = 0; pair < (population.Length / 2) - 1; pair++)
        {
            var c1 = Selection.RankSelection(population, random);
            var c2 = Selection.RankSelection(population, random);
            var c1Prime = c1.OX(c2, problem);
            var c2Prime = c2.OX(c1, problem);
            populationPrime.Add(c1Prime.InversionMutation(problem));
            populationPrime.Add(c2Prime.InversionMutation(problem));
        }

        populationPrime.AddRange(ApplyElitism(population, 10));
        population = populationPrime.ToImmutableArray();
    }
    var best = population.OrderBy(c => c.Fitness).First();
    Console.WriteLine($"[T-{id}] Best fitness: {best.Fitness}");

    return best;
}

static ImmutableArray<Chromosome> InitPopulation(double geneMultiplier, IProblem problem)
{
    var population = new List<Chromosome>();
    var random = new Random();
    var nodesCount = problem.NodeProvider.GetNodes().Count;
    var populationSize = (int)(geneMultiplier * problem.NodeProvider.GetNodes().Count);
    for (var i = 0; i < populationSize; i++)
    {
        var chromosome = Chromosome.Create(problem, random);
        population.Add(chromosome);
    }

    return population.ToImmutableArray();
}

static List<Chromosome> ApplyElitism(IEnumerable<Chromosome> population, int elitismCount)
{
    var sortedPopulation = population.OrderBy(c => c.Fitness).ToList();
    var elites = sortedPopulation.Take(elitismCount).ToList();
    return elites;
}