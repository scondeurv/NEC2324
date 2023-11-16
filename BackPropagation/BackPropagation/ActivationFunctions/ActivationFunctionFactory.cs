namespace BackPropagation.ActivationFunctions;

public class ActivationFunctionFactory
{
    public IActivationFunction Create(ActivationFunctionType type)
        => type switch
        {
            ActivationFunctionType.Linear => new Linear(),
            ActivationFunctionType.Sigmoid => new Sigmoid(),
            ActivationFunctionType.Tanh => new Tahn(),
            ActivationFunctionType.ReLu => new ReLu(),
            _ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
        };
}