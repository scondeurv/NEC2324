namespace BackPropagation.ActivationFunctions;

public sealed class ReLu : IActivationFunction
{
    public double Eval(double input) => Math.Max(0, input);
}