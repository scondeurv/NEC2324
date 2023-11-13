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
                loadedData.Add(result.Select(d => double.Parse(d, CultureInfo.InvariantCulture)).ToArray());
            }
        }

        Data = loadedData.ToArray();
    }
}