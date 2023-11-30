using LibSVMsharp;
using Tools.Common;

namespace SupportVectorMachines.RunSVM.Extensions;

public static class DatasetExtensions
{
    public static SVMProblem ToSVMProblem(this Dataset dataset)
    {
        var data = dataset.Data; 
        var datasetLength = data[data.Keys.First()].Count();
        var outputFeature = data.Keys.Last();
        var problem = new SVMProblem();
        for (var row = 0; row < datasetLength; row++)
        {
            var x = new SVMNode[data.Keys.Count() - 1];
            var index = 0;
            foreach (var feature in data.Keys)
            {
                if (feature == outputFeature)
                {
                    problem.Add(x, data[feature][row]);
                }
                else
                {
                    x[index] = new SVMNode(index + 1, data[feature][row]);
                    index++;
                }
            }
        }
        
        return problem;
    } 
}