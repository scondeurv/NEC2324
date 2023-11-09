namespace BackPropagation.ActivationFunctions;

public sealed class Tahn : IActivationFunction
{
    public double Eval(double input) => Math.Tanh(input);

    public double Derivative(double input)
    {
        var tanh = Eval(input);
        return 1 - tanh * tanh;
    }
}