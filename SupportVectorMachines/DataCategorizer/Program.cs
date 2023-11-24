if (args.Length < 1)
{
    Console.WriteLine("Please provide an input file.");
    return;
}

var inputFile = args[0];
var delimiter = ",";

if (args.Length >= 2)
{
   delimiter = args[1];
}

string[] features = default;
var mappings = new Dictionary<string, Dictionary<string, int>>();
var isHeader = true;
var dataset = File.ReadLinesAsync(inputFile);
await using var outputFile = File.CreateText($"{Path.GetFileNameWithoutExtension(inputFile)}_categorized.csv");

await foreach(var line in dataset)
{
    if (isHeader)
    {
        features = line.Split(delimiter);
        isHeader = false;
        await outputFile.WriteLineAsync(string.Join(delimiter, features));
        continue;
    }
    
    var values = line.Split(delimiter);
    for(var i = 0; i < values.Length; i++)
    {
        var feature = features[i];
        var value = values[i];
        var isNumber = double.TryParse(value, out var number);

        if (!isNumber)
        {
            if (!mappings.ContainsKey(feature))
            {
                mappings[feature] = new Dictionary<string, int>();
            }
        
            if (!mappings[feature].ContainsKey(value))
            {
                mappings[feature][value] = mappings[feature].Count;
            }
        
            values[i] = mappings[feature][value].ToString();
        }
    }
    await outputFile.WriteLineAsync(string.Join(delimiter, values));
    await outputFile.FlushAsync();
}

await using var mappingFile = File.CreateText($"{Path.GetFileNameWithoutExtension(inputFile)}_mapping.csv");
foreach(var feature in mappings)
{
    // Write the feature name as the header of the table
    await mappingFile.WriteLineAsync($"{feature.Key},Mapped Value");

    // Write each mapping as a row in the table
    foreach(var value in feature.Value)
    {
        await mappingFile.WriteLineAsync($"{value.Key},{value.Value}");
    }

    // Add an empty line to separate tables
    await mappingFile.WriteLineAsync();
}