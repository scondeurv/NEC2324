namespace BackPropagation.Scaling;

public interface IScalingMethod
{
    Task<double[]> Scale(double[] data, CancellationToken? cancellationToken = null);
}