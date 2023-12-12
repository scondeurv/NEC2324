using Accord.Statistics.Analysis;
using Accord.Statistics.Kernels;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using Tools.Common;

namespace PCA;

public sealed class PrincipalComponentAnalyzer
{
    public async Task<(PcaResult[] Results, double[] Variances)> Run(string inputFile, string delimiter, bool noHeader, CancellationToken cancellationToken = default)
    {
        var dataset = new Dataset();
        await dataset.Load(inputFile, delimiter, noHeader);
        var input = dataset
            .ToJagged()
            .Select(r => r[0..^1])
            .ToArray();
        var classes = dataset
            .ToJagged()
            .Select(r => r[^1])
            .ToArray();
        
        var pca = new PrincipalComponentAnalysis();
        pca.Learn(input);
        var output = pca
            .Transform(input);
        
        var results = new PcaResult[input.Length];
        for(var i= 0; i < output.Length; i++)
        {
            results[i] = new PcaResult
            {
                Projection = output[i].Select(o => (float)o).ToArray(),
                @class = (float)classes[i],
            };
        }
        
        var variances = Vector<double>.Build.Dense(pca.ComponentProportions);
        return (results, variances.ToArray());
    }
}