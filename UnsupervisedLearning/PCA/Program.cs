using CommandLine;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.ML;
using Microsoft.ML.Data;
using PCA;

await Parser.Default.ParseArguments<Options>(args).WithParsedAsync(async opt =>
{
    var input = File.ReadLines(opt.InputFile).First();
    var firstLine = input.Split(opt.Delimiter);
    var columns = new List<TextLoader.Column>(firstLine.Length);
    if (opt.NoHeader)
    {
        for(var i = 0; i < firstLine.Length - 1; i++)
        {
            columns.Add(new TextLoader.Column($"X{i}", DataKind.Double, i));
        }
        columns.Add(new TextLoader.Column($"Class", DataKind.Int32, firstLine.Length - 1));
    }
    else
    {
        for(var i = 0; i < firstLine.Length - 1; i++)
        {
            columns.Add(new TextLoader.Column($"{firstLine[i]}", DataKind.Double, i));
        }
        columns.Add(new TextLoader.Column($"{firstLine[^1]}", DataKind.Int32, firstLine.Length - 1));  
    }
    
    var context = new MLContext();

    var data = context.Data.LoadFromTextFile(input, hasHeader: !opt.NoHeader, separatorChar: opt.Delimiter[0], columns: columns.ToArray());
    
    var pipeline = context.Transforms
        .Concatenate("Features", columns.Take(columns.Count - 1).Select(c => c.Name ).ToArray())
        .Append(context.Transforms.ProjectToPrincipalComponents("PCA", "Features"));
    
    var model = pipeline.Fit(data);
    
    var transformedData = model.Transform(data);
    
    var transformedDataPoints = context.Data.CreateEnumerable<TransformedData>(transformedData, reuseRowObject: false);
    
    var matrix = Matrix<double>.Build.DenseOfRowArrays(transformedData.Select(x => x.PCAFeatures.Select(y => (double)y).ToArray()));
    
});