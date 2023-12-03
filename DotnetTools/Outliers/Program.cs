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
        
        if (opt.Clean != null)
        {
            var cleaner = new OutlierCleaner();
            IReadOnlyDictionary<string, double[]> cleanedData = default;
            switch (opt.Clean)
            {
                case "drop":
                    cleanedData = cleaner.DropOutliers(dataset.Data, outliers);
                    break;
                case "winsorize":
                    cleanedData = cleaner.WinsorizeOutliers(dataset.Data, outliers, 0.05, 0.9);
                    break;
                default:
                    throw new NotSupportedException(opt.Clean);
            }
            var cleanedDataset = new Dataset(cleanedData);
            var outputFileName =
                $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-cleaned{Path.GetExtension(opt.InputFile)}";
            await cleanedDataset.Save(outputFileName, opt.Delimiter, false);
        }
    });