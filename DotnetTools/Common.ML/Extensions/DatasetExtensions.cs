using Microsoft.ML;
using Microsoft.ML.Data;
using Tools.Common;

namespace Common.ML.Extensions;

public static class DatasetExtensions
{
    public static DataViewSchema GetDataViewSchema(this Dataset dataset)
    {
        var builder = new DataViewSchema.Builder();
        var features = dataset.Data.Keys;
        foreach (var feature in features)
        {
            builder.AddColumn(feature, NumberDataViewType.Double);
        }

        return builder.ToSchema();
    }
}