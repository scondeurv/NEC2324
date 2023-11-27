

using CommandLine;
using DrawDist;

await Parser.Default.ParseArguments<Options>(args)
    .WithParsedAsync(async opt => await new DatasetPlotExporter().Export(opt));