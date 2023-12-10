using System.Collections.Immutable;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace PCA;

public sealed class MLPrincipalComponentAnalyzer : IPrincipalComponentAnalyzer
{
    public Task<(PcaResult[] Results, Vector<double> Variances)> Run(string inputFile,
        string delimiter, bool noHeader, CancellationToken cancellationToken = default)
    {
        var input = File.ReadLines(inputFile).First();
        var firstLine = input.Split(delimiter);
        var columns = new List<TextLoader.Column>(firstLine.Length);
        if (noHeader)
        {
            for (var i = 0; i < firstLine.Length - 1; i++)
            {
                columns.Add(new TextLoader.Column($"X{i}", DataKind.Single, i));
            }

            columns.Add(new TextLoader.Column($"Class", DataKind.Single, firstLine.Length - 1));
        }
        else
        {
            for (var i = 0; i < firstLine.Length - 1; i++)
            {
                columns.Add(new TextLoader.Column($"{firstLine[i]}", DataKind.Single, i));
            }

            columns.Add(new TextLoader.Column($"{firstLine[^1]}", DataKind.Single, firstLine.Length - 1));
        }
        
        var context = new MLContext();
        var data = context.Data.LoadFromTextFile(inputFile, hasHeader: !noHeader, separatorChar: delimiter[0],
            columns: columns.ToArray());
        var pipeline = context.Transforms
            .Concatenate("Features", columns.Take(columns.Count - 1).Select(c => c.Name).ToArray())
            .Append(context.Transforms.ProjectToPrincipalComponents("Projection", "Features", rank: columns.Count - 1));

        var model = pipeline.Fit(data);

        var transformedData = model.Transform(data);

        var pcaResults = context.Data
            .CreateEnumerable<PcaResult>(transformedData, reuseRowObject: false)
            .OrderBy(r => r.@class)
            .ToArray();
        
        var dataMatrix = pcaResults
            .GroupBy(r => r.@class)
            .OrderBy(g => g.Key)
            .SelectMany(group => group
                .Select(p => p.Projection.Select(d => (double)d)))
            .Select(p => p.ToArray())
            .ToArray();
            
        var matrix = DenseMatrix.OfRowArrays(dataMatrix);

        var centeredMatrix = DenseMatrix.OfRows(matrix.EnumerateRows()
            .Select(row => row - row.Average()));

        var svd = centeredMatrix.Svd(true);

        var variances = svd.S;

        return Task.FromResult((pcaResults, variances));
    }
}