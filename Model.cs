using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLRealise
{
    public class Program
    {
        public class DataPoint {
            [LoadColumn(0)]
            public float Feature1 {get; set;}
            [LoadColumn(1)]
            public float Feature2 {get; set;}
            [LoadColumn(2)]
            public bool Label {get; set;}
        }
      public class Prediction{
        [ColumnName("score")]
        public float Score {get; set;}
        [ColumnName("probablity")]
        public float probablity {get; set;}

      }

      static void EvaluateMetrics(string modelName, BinaryClassificationMetrics metrics){
        Console.WriteLine($"{modelName} -  accuracy: {metrics.Accuracy:0.##}");
         Console.WriteLine($"{modelName} -  AUC: {metrics.AreaUnderRocCurve:0.##}");
        
      }
      static void Main(string[] args){
        var context = new MLContext();
        var data = context.Data.LoadFromTextFile<DataPoint>("data.csv,",separatorChar:',',hasHeader: true);
        var trainTestSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);
        var LogisticRegressionPipeline = context.Transforms.Concatenate("features","feature1","feature2")
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label",maximumNumberOfIterations:100));
        var fastTreePipeline = context.Transforms.Concatenate("features","feature1","feature2")
            .Append(context.BinaryClassification.Trainers.FastTree(labelColumnName: "label",numberOfLeaves: 50, numberOfTrees: 100));
        Console.WriteLine("Training FastTree Model...");
        var fastTreeModel = fastTreePipeline.Fit(trainTestSplit.TrainSet);
         Console.WriteLine("Evaluvating Logistic Regression model");
        var logisticRegressionPredictions = logisticRegressionModel.Transform(trainTestSplit.TrainSet);
        var logisticRegressionMetrics = context.BinaryClassification.Evaluate(logisticRegressionPredictions);

        EvaluateMetrics("Logistic Regression", logisticRegressionMetrics);

        Console.WriteLine("evakuation fast tree model..");
        var fastTreePredctions = fastTreeModel.Transform(trainTestSplit.TestSet);
        var fastTreeMetric = context.BinaryClassification.Evaluate(fastTreePredctions);

        EvaluateMetrics("FastTree", fastTreeMetric);

        if(logisticRegressionMetrics.Accuracy > fastTreeMetric.Accuracy){
          Console.WriteLine("Logistic Regression is the bzest model...");
        } else if (logisticRegressionMetrics.Accuracy < fastTreeMetric.Accuracy) {
            Console.WriteLine("fast tree is the best  model...");
        }else {
                      Console.WriteLine("Both are  best  model...");

        }

      }
    }
}