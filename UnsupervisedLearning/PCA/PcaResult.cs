using Microsoft.ML.Data;

namespace PCA;

public class PcaResult
{
    [VectorType(3)]
    public float[] Projection { get; set; }
    public float @class { get; set; }
}