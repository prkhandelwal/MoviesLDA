using MoviesCL.Services;
using System;
using System.IO;
using System.Threading.Tasks;

namespace MoviesML
{
    class Program
    {
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "data.csv");
        static async Task Main(string[] args)
        {
            Console.WriteLine("Press 1 to train and save");
            Console.WriteLine("Press 2 to predict");
            int.TryParse(Console.ReadLine(), out int input);
            switch (input)
            {
                case 1:
                    await TrainAndSave.RunTrainer(_dataPath, Environment.CurrentDirectory);
                    break;
                case 2:
                    Console.WriteLine("Insert the description text");
                    var description = Console.ReadLine();
                    //await PredictClusters.LoadModelAndPredict(description);
                    break;
                default:
                    break;
            }
        }
    }
}
