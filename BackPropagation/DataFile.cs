using System.Globalization;
using System.Text.RegularExpressions;

namespace BackPropagation;

public class DataFile
{
    private const string CsvExtension = @".csv";

    private const string CsvDelimiter = @",";
    private const string OtherDelimiter = @"\s+";

    public string[] Features { get; private set; }
    public double[][] Data { get; private set; }

    public DataFile()
    {
    }

    public DataFile(string[] features, double[][] data)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(data);
        
        Features = features;
        Data = data;
    }
    
    public async Task Load(string fileName, CancellationToken? cancellationToken = null)
    {
        var extension = Path.GetExtension(fileName);
        var delimiter = extension.Equals(CsvExtension, StringComparison.InvariantCultureIgnoreCase)
            ? CsvDelimiter
            : OtherDelimiter;

        var loadedData = new List<double[]>();
        var lines = File.ReadLinesAsync(fileName);

        var isHeader = true;
        var regex = new Regex(delimiter);
        await foreach (var line in lines)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            
            if (isHeader)
            {
                Features = regex.Split(line);
                isHeader = false;
            }
            else
            {
                var result = regex.Split(line);
                loadedData.Add(result.Select(d => double.Parse(d, NumberStyles.Float, CultureInfo.InvariantCulture)).ToArray());
            }
        }

        Data = loadedData.ToArray();
    }
    
    public async Task Save(string fileName, CancellationToken? cancellationToken = null)
    {
        var extension = Path.GetExtension(fileName);
        var delimiter = extension.Equals(CsvExtension, StringComparison.InvariantCultureIgnoreCase)
            ? CsvDelimiter
            : "\t";

        var lines = new List<string>();
        lines.Add(string.Join(delimiter, Features));
        foreach (var row in Data)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            
            lines.Add(string.Join(delimiter, row.Select(d => d.ToString("F16", CultureInfo.InvariantCulture).Replace(",", "."))));
        }

        await File.WriteAllLinesAsync(fileName, lines);
    }
}