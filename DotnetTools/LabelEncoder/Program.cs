using System.Globalization;
using CommandLine;
using Tools.Common;
using Tools.LabelEncoder;

await Parser.Default.ParseArguments<Options>(args)
    .WithParsedAsync(async opt =>
    {
        string[] features = default;
        var mappings = new Dictionary<string, Dictionary<string, double>>();
        var nonNumeric = new HashSet<string>();
        var missing = new Dictionary<string, List<int>>();
        var data = new Dictionary<string, List<string>>();
        var isHeader = true;
        var lines = File.ReadLinesAsync(opt.InputFile);

        var row = 0;
        await foreach (var line in lines)
        {
            if (isHeader)
            {
                features = line.Split(opt.Delimiter);
                isHeader = false;
                continue;
            }

            var values = line.Split(opt.Delimiter);
            for (var i = 0; i < values.Length; i++)
            {
                var feature = features[i];
                var value = values[i];
                var isNumber = double.TryParse(value, out _);

                if (!isNumber)
                {
                    if (!mappings.ContainsKey(feature))
                    {
                        mappings[feature] = new Dictionary<string, double>();
                        missing[feature] = new List<int>();
                    }

                    if (opt.MissingLabels.Contains(value))
                    {
                        missing[feature].Add(row);
                    }
                    else
                    {
                        nonNumeric.Add(feature);
                        if (!mappings[feature].ContainsKey(value))
                        {
                            mappings[feature][value] = mappings[feature].Count;
                        }

                        values[i] = mappings[feature][value].ToString(CultureInfo.InvariantCulture);
                    }
                }
                if (!data.ContainsKey(feature))
                {
                    data[feature] = new List<string>();
                }
            }

            var index = 0;
            foreach (var feature in features)
            {
                data[feature].Add(values[index++]);       
            }

            row++;
        }

        if (opt.MissingLabels.Any())
        {
            foreach (var feature in missing.Keys)
            {
                var rows = missing[feature];
                var known = data[feature].Where(v => !opt.MissingLabels.Contains(v)).Select(double.Parse).ToList();
                known.Sort();
                var value = opt.ImputationMethod switch
                {
                    "mean" => known.Average(),
                    "mode" => known.GroupBy(n => n).OrderByDescending(g => g.Count()).First().Key,
                    "median" => known.Count % 2 == 0
                        ? (known[data.Count / 2 - 1] + known[data.Count / 2]) / 2
                        : known[data.Count / 2],
                    _ => throw new NotSupportedException(opt.ImputationMethod),
                };
                
                foreach (var r in rows)
                {
                    mappings[feature][data[feature][r]] = nonNumeric.Contains(feature) ? (int)value : value;
                    data[feature][r] = mappings[feature][data[feature][r]].ToString(CultureInfo.InvariantCulture);
                }
            }
        }

        var fileName = $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-categorized.csv";
        var dataset = new Dataset(data.ToDictionary(d => d.Key,
            d => d.Value.Select(v => double.Parse(v, CultureInfo.InvariantCulture)).ToArray()));
        await dataset.Save(fileName, opt.Delimiter);

        await using var mappingFile = File.CreateText($"{Path.GetFileNameWithoutExtension(opt.InputFile)}_mapping.csv");
        foreach (var feature in mappings)
        {
            await mappingFile.WriteLineAsync($"{feature.Key},Mapped Value");

            foreach (var value in feature.Value)
            {
                await mappingFile.WriteLineAsync($"{value.Key},{value.Value}");
            }

            await mappingFile.WriteLineAsync();
        }
    });