namespace BackPropagation.Extensions;

public static class RandomExtensions
{
    public static double NextDouble(this Random rand, double min, double max)
        => rand.NextDouble() * (max - min) + min;
}