using CommandLine;
using Tensorflow.NumPy;
using Tools.Common;

namespace SupportVectorMachines.BP.Extensions;

public static class DatasetExtensions
{

    
    public static (NDArray X, NDArray Y) ToNDArrays(this Dataset dataset)
    {
        var (X, Y) = dataset.SplitMultidimensional();
        
        var xCast = new float[X.GetLength(0), X.GetLength(1)];
        for (var i = 0; i < X.GetLength(0); i++)
        {
            for (int j = 0; j < X.GetLength(1); j++)
            {
                xCast[i, j] = (float)X[i, j];
            }
        }
        
        var yCast = Y.Select(y => (float)y).ToArray();
        
        var x = new NDArray(xCast);
        var y = new NDArray(yCast);

        return (x, y);
    }
}