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
        foreach (var item in histogramSeries)
        {
            var plotModel = new PlotModel
            {
                Title = item.Key,
                Background = OxyColors.White,
            };
            
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Value" });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Count"});
            plotModel.Series.Add(item.Value);
            PngExporter.Export(plotModel, $"{item.Key}.png", 600, 400);
        }
    });

IReadOnlyDictionary<string, HistogramSeries> CreateHistogramSeries(IReadOnlyDictionary<string, double[]> dataset)
{
    var histogramSeries = new Dictionary<string, HistogramSeries>(dataset.Keys.Count());
    foreach (var feature in dataset.Keys)
    {
        var series = new HistogramSeries();
        var values = dataset[feature];
        var mean = CalculateMean(values);
        var stdDev = CalculateStandardDeviation(mean, values);
        var min = values.Min();
        var max = values.Max();
        var range = max - min;
        var binAmount = range / stdDev;
        var binSize = stdDev;

        for (var i = 0; i < binAmount; i++)
        {
            var binStart = min + (i * binSize) ;
            var binEnd = min + ((i+1) * binSize);
            var count = values.Count(v => v >= binStart && v < binEnd);
            var bin = new HistogramItem(binStart, binEnd, 1, count);
            series.Items.Add(bin);
        }

        histogramSeries.Add(feature, series);
    }

    return histogramSeries;
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

static double CalculateMean(double[] data)
{
    var acc = data.Sum();

    var mean = acc / data.Length;
    return mean;
}

static double CalculateStandardDeviation(double mean, double[] data)
{
    var acc = data.Sum(value => Math.Pow(value - mean, 2));

    var standardDeviation = Math.Sqrt(acc / data.Length);
    return standardDeviation;
}