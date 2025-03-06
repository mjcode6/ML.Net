using Microsoft.ML;
using Microsoft.ML.Data;

public class HousingData
{
    [LoadColumn(0)]
    public float SquareFeet { get; set; }

    [LoadColumn(1)]
    public float Bedrooms { get; set; }

    [LoadColumn(2)]
    public float Price { get; set; }
}

public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}

public class Program
{
    static void Main(string[] arg)
    {
        MLContext mlContext = new MLContext();
        IDataView data = mlContext.Data.LoadFromTextFile<HousingData>(
            @"C:\Users\mjsiv\Desktop\ml.Net\MLRealise\housing-data.csv",
            separatorChar: ',',
            hasHeader: true
        );

        string[] featuresColumns = { "SquareFeet", "Bedrooms" };
        string labelColumn = "Price"; // Corrigé (respecte la casse)

        var pipeline = mlContext.Transforms.Concatenate("Features", featuresColumns)
            .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: labelColumn));

        var model = pipeline.Fit(data);
        var prediction = model.Transform(data);
        var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: labelColumn);

        Console.WriteLine($"Mean absolute error: {metrics.MeanAbsoluteError}");
        Console.WriteLine($"Root mean squared error: {metrics.RootMeanSquaredError}");
    }
}
