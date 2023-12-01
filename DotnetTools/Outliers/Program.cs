using CommandLine;
using Tools.Common;
using Tools.Outliers;

await Parser.Default.ParseArguments<Options>(args)
    .WithParsedAsync(async opt =>
    {
        var dataset = new Dataset();
        await dataset.Load(opt.InputFile, opt.Delimiter, opt.NoHeader);
        
        var detector = new OutlierDetector();
        IReadOnlyDictionary<string, double[]> outliers;
        switch (opt.Method)
        {
            case "zscore":
                outliers = detector.FindOutliersWithZScore(dataset.Data, opt.Threshold);
                break;  
            case "iqr":
                outliers = detector.FindOutliersWithIQR(dataset.Data);
                break;
            default:
                throw new NotSupportedException(opt.Method);
        }
        
        detector.ExportOutliers(dataset.Data, outliers);
        
        if (opt.Clean)
        {
            var cleaner = new OutlierCleaner();
            var cleanedData = cleaner.CleanOutliers(dataset.Data, outliers);
            var cleanedDataset = new Dataset(cleanedData);
            var outputFileName =
                $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-cleaned{Path.GetExtension(opt.InputFile)}";
            await cleanedDataset.Save(outputFileName, opt.Delimiter, false);
        }
    });