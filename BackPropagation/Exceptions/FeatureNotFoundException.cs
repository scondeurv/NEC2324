namespace BackPropagation.Exceptions;

public class FeatureNotFoundException : Exception
{
    public FeatureNotFoundException(string feature) : base($"Feature {feature} is not found")
    {
    }
}