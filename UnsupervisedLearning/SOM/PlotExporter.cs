using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Core.Drawing;
using OxyPlot.Series;

namespace SOM;

public class PlotExporter
{
    public void PlotHeatmap(double[,] data, string outputPath, int width, int height, string title)
    {
        var model = new PlotModel { Title = title};

        var heatMapSeries = new HeatMapSeries
        {
            X0 = 0,
            X1 = width,
            Y0 = 0,
            Y1 = height,
            Interpolate = false,
            RenderMethod = HeatMapRenderMethod.Bitmap,
            Data = data
        };

        model.Series.Add(heatMapSeries);
        model.Background = OxyColors.White;
        model.Axes.Add(new LinearColorAxis { Position = AxisPosition.Right, Palette = OxyPalettes.Jet(200)  });

        var pngExporter = new PngExporter { Width = 600, Height = 400 };
        pngExporter.ExportToFile(model, outputPath);
    }
    
    public void PlotHeatmap(Dictionary<int, List<int>> clusters, string outputPath, int width, int height, string title)
    {
        var model = new PlotModel { Title = title };
        
        var heatmapData = ConvertToHeatmapData(clusters, width, height);
        
        var heatMapSeries = new HeatMapSeries
        {
            X0 = 0,
            X1 = width,
            Y0 = 0,
            Y1 = height,
            Interpolate = false,
            RenderMethod = HeatMapRenderMethod.Bitmap,
            Data = heatmapData
        };

        model.Series.Add(heatMapSeries);
        model.Background = OxyColors.White;
        model.Axes.Add(new LinearColorAxis { Position = AxisPosition.Right, Palette = OxyPalettes.Jet(200)  });

        var pngExporter = new PngExporter { Width = 600, Height = 400 };
        pngExporter.ExportToFile(model, outputPath);
    }
    
    public void PlotUMatrix(double[,] uMatrix, string outputPath, int width, int height, string title)
    {
        var model = new PlotModel { Title = title };

        var heatMapSeries = new HeatMapSeries
        {
            X0 = 0,
            X1 = width,
            Y0 = 0,
            Y1 = height,
            Interpolate = false,
            RenderMethod = HeatMapRenderMethod.Bitmap,
            Data = uMatrix
        };

        model.Series.Add(heatMapSeries);
        model.Background = OxyColors.White;
        model.Axes.Add(new LinearColorAxis { Position = AxisPosition.Right, Palette = OxyPalettes.Gray(200).Reverse() });

        var pngExporter = new PngExporter { Width = 600, Height = 400 };
        pngExporter.ExportToFile(model, outputPath);
    }

    private static double[,] ConvertToHeatmapData(Dictionary<int, List<int>> clusters, int width, int height)
    {
        var heatmapData = new double[width, height];
        
        foreach (var cluster in clusters)
        {
            var y = Math.DivRem(cluster.Key, width, out var x);
            heatmapData[x, y] = cluster.Value.Count;
        }

        return heatmapData;
    }
}