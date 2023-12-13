using Aglomera;

namespace AHC;

public class EuclideanDistance : IDissimilarityMetric<DataPoint>
{
    public double Calculate(DataPoint p1, DataPoint p2)
    {
        var sum2 = 0d;
        var length = Math.Min(p1.Value.Length, p2.Value.Length);
        for (var idx1 = 0; idx1 < length; ++idx1)
        {
            var delta = p1.Value[idx1] - p2.Value[idx1];
            sum2 += delta * delta;
        }

        return Math.Sqrt(sum2);
    }
}