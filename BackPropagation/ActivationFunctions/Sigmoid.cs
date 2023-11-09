namespace BackPropagation.ActivationFunctions;

public sealed class Sigmoid : IActivationFunction
{
    public double Eval(double input) => 1.0 / (1.0 + Math.Exp(-input));
}