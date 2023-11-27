using AutoFixture.Xunit2;

namespace DrawDist.Tests;

public class DatasetPlotExporterTests
{
    private readonly DatasetPlotExporter _exporter = new();
    
    [Theory]
    [InlineAutoData("Datasets/A2-ring-merged.txt", "\t", true)]
    [InlineAutoData("Datasets/A2-ring-separable.txt", "\t", true)]
    [InlineAutoData("Datasets/A2-ring-test.txt", "\t", true)]
    [InlineAutoData("Datasets/bank-additional-full_categorized.csv", ";", false)]
    public async Task CreatePlots(string inputFile, string delimiter, bool noHeader)
    {
        //Arrange
        var options = new Options
        {
            InputFile = inputFile,
            Delimiter = delimiter,
            NoHeader = noHeader
        };
        
        //Act
        await _exporter.Export(options);
    }
}