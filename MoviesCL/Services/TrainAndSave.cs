using Microsoft.ML;
using MoviesCL.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MoviesCL.Services
{
    public class TrainAndSave
    {
        public static async Task RunTrainer(string _dataPath, string rootDir)
        {
            var mlContext = new MLContext();

            var dataview = mlContext.Data.LoadFromTextFile<TextData>(_dataPath, hasHeader: true, separatorChar: ',');

            IEnumerable<TextData> dataEnumerable = mlContext.Data.CreateEnumerable<TextData>(dataview, reuseRowObject: true).Where(item => !string.IsNullOrWhiteSpace(item.Plot));
            dataview = mlContext.Data.LoadFromEnumerable(dataEnumerable);

            // A pipeline for featurizing the text/string using 
            // LatentDirichletAllocation API. o be more accurate in computing the
            // LDA features, the pipeline first normalizes text and removes stop
            // words before passing tokens (the individual words, lower cased, with
            // common words removed) to LatentDirichletAllocation.

            var pipeline = mlContext.Transforms.Text.NormalizeText("NormalizedText",
                "Plot")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens",
                    "NormalizedText"))
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Tokens"))
                .Append(mlContext.Transforms.Text.ProduceNgrams("Tokens", ngramLength: 3))
                .Append(mlContext.Transforms.Text.LatentDirichletAllocation(
                    "Features", "Tokens", numberOfTopics: 10));

            // Fit to data.
            var transformer = pipeline.Fit(dataview);
            mlContext.Model.Save(transformer, dataview.Schema, "model.zip");
            // Create the prediction engine to get the LDA features extracted from
            // the text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, PredictionData>(transformer);

            List<PredictionData> predictions = new List<PredictionData>();
            Console.WriteLine(dataEnumerable.Count());

            foreach (var item in dataEnumerable)
            {
                var pred = predictionEngine.Predict(item);
                predictions.Add(pred);
                Console.WriteLine(JsonConvert.SerializeObject(pred));
                Console.WriteLine();
            }

            //await AddPredictionsToDb(predictions);
            //Console.WriteLine("Database Updated");
            Console.ReadLine();
        }
    }
}
