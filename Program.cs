using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

class Program
{
  private const int GridSize = 10;
  private static bool[,] grid = new bool[GridSize, GridSize];
  private static SimpleNeuralNetwork? neuralNetwork; 
  private static readonly string NetworkFilePath = "network_data.json";

  static void Main()
  {
    Console.WriteLine("Digit recognition application");
    Console.WriteLine("=============================");

    if (File.Exists(NetworkFilePath))
    {
      try
      {
        LoadNetwork();
        Console.WriteLine("Neural network loaded from file.");
      }
      catch (Exception ex)
      {
        Console.WriteLine($"Network loading error: {ex.Message}");
        CreateAndTrainNetwork();
      }
    }
    else
    {
      CreateAndTrainNetwork();
    }

    while (true)
    {
      Console.Clear(); 
      DrawGrid();

      Console.WriteLine("\nInstructions:");
      Console.WriteLine("- Press 'c' to clear the grid");
      Console.WriteLine("- Press 'd' to detect the digit");
      Console.WriteLine("- Press 's' to save the network");
      Console.WriteLine("- Press 't' to retrain the network");
      Console.WriteLine("- Press 'q' to exit.");
      Console.WriteLine("- Enter coordinates (ex: '2,2 3,4') to activate cells");

      var input = Console.ReadLine()?.Trim().ToLower() ?? ""; 

      if (input == "q") break;
      else if (input == "c") ClearGrid();
      else if (input == "d") DetectDigit();
      else if (input == "s") SaveNetwork();
      else if (input == "t") 
      {
        CreateAndTrainNetwork();
        SaveNetwork();
      }
      else
      {
        bool hasError = false;
        try
        {
          var coordinatePairs = input.Split(' ');
          foreach (var pair in coordinatePairs)
          {
            var parts = pair.Split(',');
            if (parts.Length != 2) throw new FormatException();

            int col = int.Parse(parts[0]);
            int row = int.Parse(parts[1]);

            if (row >= 0 && row < GridSize && col >= 0 && col < GridSize)
            {
              grid[row, col] = true;
            }
            else
            {
              Console.WriteLine($"Invalid coordinates: {pair}. Ignored.");
              hasError = true;
            }
          }
        }
        catch
        {
          Console.WriteLine("Invalid format! Use 'x,y' or 'x,y x,y'.");
          hasError = true;
        }

        if (hasError)
        {
          Console.WriteLine("Press ENTER to continue...");
          Console.ReadLine(); 
        }
      }
    }
  }

  static void CreateAndTrainNetwork()
  {
    neuralNetwork = new SimpleNeuralNetwork(GridSize * GridSize, 300, 10);
    TrainNetwork();
  }

  static void ClearGrid()
  {
    for (int i = 0; i < GridSize; i++)
      for (int j = 0; j < GridSize; j++)
        grid[i, j] = false;
  }

  static void DrawGrid()
  {
    Console.WriteLine("  0 1 2 3 4 5 6 7 8 9");
    for (int i = 0; i < GridSize; i++)
    {
      Console.Write(i + " ");
      for (int j = 0; j < GridSize; j++)
      {
        Console.Write(grid[i, j] ? "■ " : "□ ");
      }
      Console.WriteLine();
    }
  }

  static void DetectDigit()
  {
    if (neuralNetwork == null)
    {
      Console.WriteLine("Error: Neural network is not initialized!");
      Console.WriteLine("Press ENTER to continue...");
      Console.ReadLine();
      return;
    }

    double[] input = new double[GridSize * GridSize];
    int index = 0;

    for (int i = 0; i < GridSize; i++)
      for (int j = 0; j < GridSize; j++)
        input[index++] = grid[i, j] ? 1.0 : 0.0;

    double[] output = neuralNetwork.FeedForward(input);
    
    Console.WriteLine("\nConfidentiality for all figures:");
    for (int i = 0; i < 10; i++)
    {
      Console.WriteLine($"Digit {i}: {output[i] * 100:F1}%");
    }
    
    int recognizedDigit = Array.IndexOf(output, output.Max());
    double confidence = output[recognizedDigit] * 100;

    Console.WriteLine($"\nRecognized digit: {recognizedDigit} (Confidence: {confidence:F1}%)");
    Console.WriteLine("Press ENTER to continue...");
    Console.ReadLine();
  }

  static void TrainNetwork()
  {
    if (neuralNetwork == null)
    {
      Console.WriteLine("Error: Neural network is not initialized!");
      return;
    }

    Console.WriteLine("I'm training the neural network...");

    for (int epoch = 0; epoch < 5000; epoch++)
    {
      if (epoch % 100 == 0)
        Console.WriteLine($"Epoch: {epoch}/5000");

      for (int digit = 0; digit < 10; digit++)
      {
        double[] target = new double[10];
        target[digit] = 1.0;

        double[] basePattern = new double[GridSize * GridSize];
        CreateDigitPattern(digit, basePattern);
        neuralNetwork.Train(basePattern, target);

        if (epoch % 10 == 0)
        {
          double[] noisyPattern = AddNoise(basePattern, 0.05);
          neuralNetwork.Train(noisyPattern, target);

          double[] shiftedPatternRight = ShiftPattern(basePattern, 1, 0);
          neuralNetwork.Train(shiftedPatternRight, target);

          double[] shiftedPatternLeft = ShiftPattern(basePattern, -1, 0);
          neuralNetwork.Train(shiftedPatternLeft, target);

          double[] shiftedPatternUp = ShiftPattern(basePattern, 0, -1);
          neuralNetwork.Train(shiftedPatternUp, target);

          double[] shiftedPatternDown = ShiftPattern(basePattern, 0, 1);
          neuralNetwork.Train(shiftedPatternDown, target);
        }
      }
    }

    Console.WriteLine("Training completed!");
  }

  static double[] AddNoise(double[] pattern, double noiseLevel)
  {
    double[] noisyPattern = new double[pattern.Length];
    Array.Copy(pattern, noisyPattern, pattern.Length);
    
    Random random = new Random();
    int pixelsToFlip = (int)(pattern.Length * noiseLevel);
    
    for (int i = 0; i < pixelsToFlip; i++)
    {
      int index = random.Next(0, pattern.Length);
      noisyPattern[index] = noisyPattern[index] == 1.0 ? 0.0 : 1.0;
    }
    
    return noisyPattern;
  }

  static double[] ShiftPattern(double[] pattern, int shiftX, int shiftY)
  {
    double[] shiftedPattern = new double[pattern.Length];
    
    Array.Fill(shiftedPattern, 0.0);
    
    for (int y = 0; y < GridSize; y++)
    {
      for (int x = 0; x < GridSize; x++)
      {
        int newX = x + shiftX;
        int newY = y + shiftY;
        
        if (newX >= 0 && newX < GridSize && newY >= 0 && newY < GridSize)
        {
          int oldIndex = y * GridSize + x;
          int newIndex = newY * GridSize + newX;
          
          shiftedPattern[newIndex] = pattern[oldIndex];
        }
      }
    }
    
    return shiftedPattern;
  }

  static void SaveNetwork()
  {
    if (neuralNetwork == null)
    {
      Console.WriteLine("Error: No network to save.");
      Console.WriteLine("Press ENTER to continue...");
      Console.ReadLine();
      return;
    }

    try
    {
      var options = new JsonSerializerOptions
      {
        WriteIndented = true
      };
      string jsonString = JsonSerializer.Serialize(neuralNetwork, options);
      File.WriteAllText(NetworkFilePath, jsonString);
      
      Console.WriteLine("Neural network successfully saved!");
      Console.WriteLine("Press ENTER to continue...");
      Console.ReadLine();
    }
    catch (Exception ex)
    {
      Console.WriteLine($"Error saving network: {ex.Message}");
      Console.WriteLine("Press ENTER to continue...");
      Console.ReadLine();
    }
  }

  static void LoadNetwork()
  {
    try
    {
      string jsonString = File.ReadAllText(NetworkFilePath);
      neuralNetwork = JsonSerializer.Deserialize<SimpleNeuralNetwork>(jsonString);
      
      if (neuralNetwork == null)
      {
        throw new Exception("Deserializarea a returnat null");
      }
    }
    catch (Exception ex)
    {
      Console.WriteLine($"Network loading error: {ex.Message}");
      Console.WriteLine("A new network will be created.");
      CreateAndTrainNetwork();
    }
  }

  static void CreateDigitPattern(int digit, double[] pattern)
  {
    for (int i = 0; i < pattern.Length; i++)
      pattern[i] = 0;

    switch (digit)
    {
      case 0: 
        for (int i = 2; i < GridSize - 2; i++)
        {
          pattern[2 * GridSize + i] = 1; 
          pattern[(GridSize - 3) * GridSize + i] = 1; 
        }
        for (int i = 2; i < GridSize - 2; i++)
        {
          pattern[i * GridSize + 2] = 1; 
          pattern[i * GridSize + (GridSize - 3)] = 1; 
        }
        break;

      case 1: 
        for (int i = 2; i < GridSize - 2; i++)
        {
          pattern[i * GridSize + GridSize / 2] = 1; 
        }
        pattern[(GridSize - 3) * GridSize + (GridSize / 2 - 1)] = 1;
        pattern[(GridSize - 3) * GridSize + (GridSize / 2 + 1)] = 1;
        break;

      case 2: 
        for (int j = 3; j < GridSize - 3; j++)
        {
          pattern[2 * GridSize + j] = 1; 
          pattern[5 * GridSize + j] = 1; 
          pattern[8 * GridSize + j] = 1; 
        }
        for (int i = 2; i < 5; i++)
          pattern[i * GridSize + (GridSize - 4)] = 1; 
        for (int i = 5; i < 8; i++)
          pattern[i * GridSize + 3] = 1; 
        break;

      case 3: 
        for (int j = 3; j < GridSize - 3; j++)
        {
          pattern[2 * GridSize + j] = 1; 
          pattern[5 * GridSize + j] = 1; 
          pattern[8 * GridSize + j] = 1; 
        }
        for (int i = 2; i < 8; i++)
          if (i != 5)
            pattern[i * GridSize + (GridSize - 4)] = 1; 
        break;

      case 4: 
        for (int i = 2; i < 6; i++)
          pattern[i * GridSize + 3] = 1; 
        for (int j = 3; j < GridSize - 3; j++)
          pattern[5 * GridSize + j] = 1; 
        for (int i = 2; i < 9; i++)
          pattern[i * GridSize + (GridSize - 4)] = 1; 
        break;

      case 5: 
        for (int j = 3; j < GridSize - 3; j++)
        {
          pattern[2 * GridSize + j] = 1; 
          pattern[5 * GridSize + j] = 1; 
          pattern[8 * GridSize + j] = 1; 
        }
        for (int i = 2; i < 5; i++)
          pattern[i * GridSize + 3] = 1; 
        for (int i = 5; i < 8; i++)
          pattern[i * GridSize + (GridSize - 4)] = 1; 
        break;

      case 6: 
        for (int j = 3; j < GridSize - 3; j++)
        {
          pattern[2 * GridSize + j] = 1; 
          pattern[5 * GridSize + j] = 1; 
          pattern[8 * GridSize + j] = 1; 
        }
        for (int i = 2; i < 8; i++)
          pattern[i * GridSize + 3] = 1; 
        for (int i = 5; i < 8; i++)
          pattern[i * GridSize + (GridSize - 4)] = 1; 
        break;

      case 7: 
        for (int j = 3; j < GridSize - 3; j++)
          pattern[2 * GridSize + j] = 1; 
        for (int i = 2; i < 9; i++)
          pattern[i * GridSize + (GridSize - 4)] = 1; 
        break;

      case 8: 
        for (int j = 3; j < GridSize - 3; j++)
        {
          pattern[2 * GridSize + j] = 1; 
          pattern[5 * GridSize + j] = 1; 
          pattern[8 * GridSize + j] = 1; 
        }
        for (int i = 2; i < 8; i++)
          if (i != 5)
          {
            pattern[i * GridSize + 3] = 1; 
            pattern[i * GridSize + (GridSize - 4)] = 1; 
          }
        break;

      case 9: 
        for (int j = 3; j < GridSize - 3; j++)
        {
          pattern[2 * GridSize + j] = 1; 
          pattern[5 * GridSize + j] = 1; 
          pattern[8 * GridSize + j] = 1; 
        }
        for (int i = 2; i < 5; i++)
          pattern[i * GridSize + 3] = 1; 
        for (int i = 2; i < 9; i++)
          pattern[i * GridSize + (GridSize - 4)] = 1; 
        break;
    }
  }
}

[Serializable]
public class SimpleNeuralNetwork
{
  [JsonInclude]
  public int InputSize { get; private set; }
  
  [JsonInclude]
  public int HiddenSize { get; private set; }
  
  [JsonInclude]
  public int OutputSize { get; private set; }

  [JsonInclude]
  public double[][] WeightsInputHidden { get; private set; }
  
  [JsonInclude]
  public double[][] WeightsHiddenOutput { get; private set; }
  
  [JsonInclude]
  public double[] BiasHidden { get; private set; }
  
  [JsonInclude]
  public double[] BiasOutput { get; private set; }

  private readonly Random random = new Random();
  private readonly double learningRate = 0.005;

  public SimpleNeuralNetwork() 
  {
    InputSize = 100;
    HiddenSize = 300;
    OutputSize = 10;
    InitializeArrays();
  }

  public SimpleNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
  {
    InputSize = inputSize;
    HiddenSize = hiddenSize;
    OutputSize = outputSize;

    InitializeArrays();

    double inputScale = Math.Sqrt(2.0 / (inputSize + hiddenSize));
    double hiddenScale = Math.Sqrt(2.0 / (hiddenSize + outputSize));

    for (int i = 0; i < inputSize; i++)
      for (int j = 0; j < hiddenSize; j++)
        WeightsInputHidden[i][j] = (random.NextDouble() * 2 - 1) * inputScale;

    for (int i = 0; i < hiddenSize; i++)
    {
      BiasHidden[i] = 0;
      for (int j = 0; j < outputSize; j++)
        WeightsHiddenOutput[i][j] = (random.NextDouble() * 2 - 1) * hiddenScale;
    }

    for (int i = 0; i < outputSize; i++)
      BiasOutput[i] = 0;
  }

  private void InitializeArrays()
  {
    WeightsInputHidden = new double[InputSize][];
    for (int i = 0; i < InputSize; i++)
      WeightsInputHidden[i] = new double[HiddenSize];

    WeightsHiddenOutput = new double[HiddenSize][];
    for (int i = 0; i < HiddenSize; i++)
      WeightsHiddenOutput[i] = new double[OutputSize];

    BiasHidden = new double[HiddenSize];
    BiasOutput = new double[OutputSize];
  }

  private double Sigmoid(double x)
  {
    if (x < -20.0) return 0.0;
    if (x > 20.0) return 1.0;
    return 1.0 / (1.0 + Math.Exp(-x));
  }

  private double SigmoidDerivative(double x)
  {
    return x * (1 - x);
  }

  public double[] FeedForward(double[] input)
  {
    double[] hiddenLayer = new double[HiddenSize];
    for (int i = 0; i < HiddenSize; i++)
    {
      double sum = BiasHidden[i];
      for (int j = 0; j < InputSize; j++)
        sum += input[j] * WeightsInputHidden[j][i];
      hiddenLayer[i] = Sigmoid(sum);
    }

    double[] outputLayer = new double[OutputSize];
    for (int i = 0; i < OutputSize; i++)
    {
      double sum = BiasOutput[i];
      for (int j = 0; j < HiddenSize; j++)
        sum += hiddenLayer[j] * WeightsHiddenOutput[j][i];
      outputLayer[i] = Sigmoid(sum);
    }

    return outputLayer;
  }

  public void Train(double[] input, double[] target)
  {
    // Forward pass
    double[] hiddenLayer = new double[HiddenSize];
    for (int i = 0; i < HiddenSize; i++)
    {
      double sum = BiasHidden[i];
      for (int j = 0; j < InputSize; j++)
        sum += input[j] * WeightsInputHidden[j][i];
      hiddenLayer[i] = Sigmoid(sum);
    }

    double[] outputLayer = new double[OutputSize];
    for (int i = 0; i < OutputSize; i++)
    {
      double sum = BiasOutput[i];
      for (int j = 0; j < HiddenSize; j++)
        sum += hiddenLayer[j] * WeightsHiddenOutput[j][i];
      outputLayer[i] = Sigmoid(sum);
    }

    double[] outputErrors = new double[OutputSize];
    for (int i = 0; i < OutputSize; i++)
      outputErrors[i] = (target[i] - outputLayer[i]) * SigmoidDerivative(outputLayer[i]);

    double[] hiddenErrors = new double[HiddenSize];
    for (int i = 0; i < HiddenSize; i++)
    {
      double error = 0;
      for (int j = 0; j < OutputSize; j++)
        error += outputErrors[j] * WeightsHiddenOutput[i][j];
      hiddenErrors[i] = error * SigmoidDerivative(hiddenLayer[i]);
    }

    for (int i = 0; i < HiddenSize; i++)
    {
      for (int j = 0; j < OutputSize; j++)
      {
        double delta = learningRate * outputErrors[j] * hiddenLayer[i];
        delta = Math.Max(-0.1, Math.Min(0.1, delta));
        WeightsHiddenOutput[i][j] += delta;
      }
    }

    for (int i = 0; i < InputSize; i++)
    {
      for (int j = 0; j < HiddenSize; j++)
      {
        double delta = learningRate * hiddenErrors[j] * input[i];
        delta = Math.Max(-0.1, Math.Min(0.1, delta));
        WeightsInputHidden[i][j] += delta;
      }
    }

    for (int i = 0; i < OutputSize; i++)
      BiasOutput[i] += learningRate * outputErrors[i];

    for (int i = 0; i < HiddenSize; i++)
      BiasHidden[i] += learningRate * hiddenErrors[i];
  }
}