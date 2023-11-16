namespace BackPropagation.ActivationFunctions;

public sealed class Sigmoid : IActivationFunction
{
    public double Eval(double input) => 1.0 / (1.0 + Math.Exp(-input));
    
    public double Derivative(double input)
    {
        var sigmoid = Eval(input);
        return sigmoid * (1 - sigmoid);
    }
}