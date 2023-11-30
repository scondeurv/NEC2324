namespace Tools.Common.Scaling;

public class LogScaler : IScaler
{
    public Task<double[]> Scale(double[] data, CancellationToken? cancellationToken = null)
    {
        return Task.FromResult(data.Select(x => Math.Log(x)).ToArray());
    }

    public Task<double[]> Descale(double[] scaledData, CancellationToken? cancellationToken = null)
    {
        return Task.FromResult(scaledData.Select(x => Math.Exp(x)).ToArray());
    }
}