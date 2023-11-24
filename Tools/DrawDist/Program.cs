using System.Data;
using System.Globalization;
using CommandLine;
using DrawDist;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Series;

await Parser.Default.ParseArguments<Options>(args)
    .WithParsedAsync<Options>(async opt =>
    {
        var dataset = await ReadFile(opt.InputFile, opt.Delimiter, opt.NoHeader);
        var histogramSeries = CreateHistogramSeries(dataset);
        var lineSeries = await CreateLineSeries(dataset);
        foreach (var item in histogramSeries)
        {
            var plotModel = new PlotModel
            {
                Title = item.Key,
                Background = OxyColors.White,
            };
            
            plotModel.Series.Add(item.Value);
            plotModel.Series.Add(lineSeries[item.Key]);
            PngExporter.Export(plotModel, $"{item.Key}.png", 600, 400);
        }
    });

IReadOnlyDictionary<string, HistogramSeries> CreateHistogramSeries(IReadOnlyDictionary<string, double[]> dataset, double binSize = 0.01)
{
    var histogramSeries = new Dictionary<string, HistogramSeries>(dataset.Keys.Count());
    foreach (var feature in dataset.Keys)
    {
        var series = new HistogramSeries();
        var values = dataset[feature];
        var min = values.Min();
        var max = values.Max();
        var range = max - min;
        var binAmount = (range / binSize) + (range % binSize > 0 ? 1 : 0);

        for (var i = 0; i < binAmount; i++)
        {
            var binStart = min + (i * binSize) ;
            var binEnd = min + ((i+1) * binSize);
            var count = values.Count(v => v >= binStart && v < binEnd);
            var bin = new HistogramItem(binStart, binEnd, binSize * count, 1);
            series.Items.Add(bin);
        }

        series.FillColor = OxyColors.Blue;
        histogramSeries.Add(feature, series);
    }

    return histogramSeries;
}

static IReadOnlyDictionary<string, LineSeries> CreateLineSeries(IReadOnlyDictionary<string, double[]> dataset)
{
    var lineSeries = new Dictionary<string, LineSeries>();
    foreach (var item in dataset)
    {
        var values = item.Value;
        var min = values.Min();
        var max = values.Max();
        var binSize = (max - min) / 100;
        var mean = values.Average();
        var stdDev = Math.Sqrt(values.Sum(v => Math.Pow(v - mean, 2)) / values.Length);
        var line = new double[100];
        for (var x = min; x <= max; x += binSize)
        {
            line[(int)((x - min) / binSize)] = NormalDistribution(x, mean, stdDev);
        }

        lineSeries.Add(item.Key, line);
    }

    return lineSeries;
}

static double NormalDistribution(double x, double mean, double stdDev)
{
    var factor = 1.0 / Math.Sqrt(2 * Math.PI * stdDev * stdDev);
    var exponent = -(x - mean) * (x - mean) / (2 * stdDev * stdDev);
    return factor * Math.Exp(exponent);
}

static async Task<IReadOnlyDictionary<string, double[]>> ReadFile(string fileName, string delimiter, bool noHeader)
{
    var data = File.ReadLinesAsync(fileName);
    var isHeader = !noHeader;
    var table = new Dictionary<string, List<double>>();
    await foreach (var row in data)
    {
        if (isHeader)
        {
            var features = row.Split(delimiter);
            foreach (var feature in features)
            {
                table.Add(feature, new List<double>());
            }

            isHeader = false;
            continue;
        }

        if (table.Count == 0)
        {
            var index = 1;
            foreach (var item in row.Split(delimiter))
            {
                table.Add($"Feature {index++}", new List<double>());
            }
        }

        var values = row
            .Split(delimiter)
            .Select(v => double.Parse(v, CultureInfo.InvariantCulture))
            .ToArray();
        var col = 0;
        foreach (var feature in table.Keys)
        {
            table[feature].Add(values[col++]);
        }
    }

    return table.ToDictionary(t => t.Key, t => t.Value.ToArray());
}