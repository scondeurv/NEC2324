using MathNet.Numerics.LinearAlgebra;

namespace PCA;

public interface IPrincipalComponentAnalyzer
{
    Task<(PcaResult[] Results, Vector<double> Variances)> Run(string inputFile,
        string delimiter, bool noHeader, CancellationToken cancellationToken = default);
}