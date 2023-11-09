namespace BackPropagation.ActivationFunctions;

public sealed class Tahn : IActivationFunction
{
    public double Eval(double input) => Math.Tanh(input);
}