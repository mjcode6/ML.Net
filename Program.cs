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

    [LoadColumn(3)]
    public float Neighborhood {get; set;}
}

public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}

public class TransformHousingData{
    public float SquareFeet {get; set;}
     public float Bedrooms {get; set;}
          public float Price {get; set;}

     public  float[] Features {get; set;}

       public  float[] Neighborhood {get; set;}


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

        var dataPipline = mlContext.Transforms.Conversion.ConvertType("SquareFeet",outputKind: DataKind.Single)
        .Append(mlContext.Transforms.NormalizeMinMax("SquareFeet"))
        .Append(mlContext.Transforms.Concatenate("Features", "SquareFeet", "Bedrooms"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"));
        var transformeData = dataPipline.Fit(data).Transform(data);
        var transformedDataEnumerable = mlContext.Data.CreateEnumerable<TransformHousingData>(transformeData, reuseRowObject: false).ToList();
        foreach(var item in transformedDataEnumerable){
            Console.WriteLine($"SquareFeet: {item.SquareFeet}, BedRooms: {item.Bedrooms}, Price: {item.Price},Features: [{string.Join(",",item.Features)}],Neiborhood:[{string.Join(",",item.Neighborhood)}]");
        }

    }
}
