namespace Tools.Common.Scaling;

public interface IScaler
{
    Task<double[]> Scale(double[] data, CancellationToken? cancellationToken = null);
    Task<double[]> Descale(double[] scaledData, CancellationToken? cancellationToken = null);
}