using DataNormalizer;

public class DatasetNormalizer
{
    public async Task Normalize(Options opt)
    {
        // Read the dataset from the input file
        var dataset = await ReadFile(opt.InputFile, opt.Delimiter);

        // Normalize the features
        foreach (var feature in opt.FeatureNames ?? dataset.Keys)
        {
            if (!dataset.ContainsKey(feature))
            {
                continue;
            }

            foreach (var method in opt.ScalingMethods)
            {
                switch (method.ToLower())
                {
                    case "zscore":
                        dataset[feature] = dataset[feature].ZScore().ToArray();
                        break;
                    case "minmax":
                        dataset[feature] = dataset[feature].Normalize(opt.MinMaxScale).ToArray();
                        break;
                }
            }
        }

        // Write the normalized dataset to the output file
        await WriteFile($"{opt.InputFile}-normalized", opt.Delimiter, dataset);
    }

    private async Task<IReadOnlyDictionary<string, double[]>> ReadFile(string fileName, string delimiter)
    {
        // Implement this method to read the dataset from the input file
        throw new NotImplementedException();
    }

    private async Task WriteFile(string fileName, string delimiter, IReadOnlyDictionary<string, double[]> dataset)
    {
        // Implement this method to write the normalized dataset to the output file
        throw new NotImplementedException();
    }
}