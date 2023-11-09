namespace BackPropagation.ActivationFunctions;

public interface IActivationFunction
{
    double Eval(double input);
    double Derivative(double input);
}