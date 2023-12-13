using Aglomera;
using Aglomera.D3;
using Aglomera.Linkage;
using AHC;
using AHC.Extensions;
using CommandLine;
using Newtonsoft.Json;
using Tools.Common;

await Parser.Default.ParseArguments<Options>(args).WithParsedAsync(async opt =>
{
    var dataset = new Dataset();
    await dataset.Load(opt.InputFile, opt.Separator, opt.NoHeader);
    var dataPoints = dataset.ToDataPoints();

    var metric = new EuclideanDistance();
    ILinkageCriterion<DataPoint> linkage = opt.Linkage switch
    {
        "upgma" => new AverageLinkage<DataPoint>(metric),
        "complete" => new CompleteLinkage<DataPoint>(metric),
        _ => throw new ArgumentException($"Invalid linkage: {opt.Linkage}")
    };
    var algorithm = new AgglomerativeClusteringAlgorithm<DataPoint>(linkage);

    var clustering = algorithm.GetClustering(dataPoints);

    var fileName = $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-clustering.json";
    clustering.SaveD3DendrogramFile(fileName, formatting: Formatting.Indented);
});