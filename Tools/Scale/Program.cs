using CommandLine;
using Tools.Common;
using Tools.Common.Scaling;
using Tools.Scale;

await Parser.Default.ParseArguments<Options>(args)
    .WithParsedAsync(async opt =>
    {
        var dataset = new Dataset();
        await dataset.Load(opt.InputFile, opt.Delimiter, opt.NoHeader);
        var factory = new ScalerFactory();
        var scalingMethodPerFeature = factory.CreatePerFeature(opt.ScalingPerFeature.Select(opt => opt.Split(':'))
            .ToDictionary(opt => opt[0], opt => opt[1]));
        await dataset.Scale(scalingMethodPerFeature);
        var outputFileName =
            $"{Path.GetFileNameWithoutExtension(opt.InputFile)}-scaled{Path.GetExtension(opt.InputFile)}";
        await dataset.Save(outputFileName, opt.Delimiter, true);
    });