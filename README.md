Digit Recognizer Console App 🧠🔢

A C# console project that implements a simple feedforward neural network for recognizing handwritten digits (0–9) drawn on a 10x10 grid. Users interact with the app via terminal input, manually activating grid cells to simulate drawing digits.

🧩 Features

🧠 Custom neural network with one hidden layer
🖼️ Draw digits using a coordinate-based 10x10 grid
🔁 Automatic training with hardcoded digit patterns
💾 Save/load the trained network as a JSON file (network_data.json)
✍️ Simple and intuitive terminal interface
🚀 How to Run

Make sure you have the .NET SDK installed (Download .NET)
Clone this repository:
git clone https://github.com/your-username/digit-recognizer.git
cd digit-recognizer
Run the application:
dotnet run
🎮 Controls

Enter coordinates in the format x,y (e.g. 2,3 4,5) to activate cells.
Available commands:
c – clear the grid
d – detect the digit
s – save the current neural network
t – retrain the network from scratch
q – quit the application
🧠 Neural Network Details

Grid size: 10x10 (100 input neurons)
Hidden layer: 300 neurons
Output layer: 10 neurons (representing digits 0–9)
Activation function: Sigmoid
Data persistence: JSON serialization
📦 Code Structure

Program.cs: CLI interface, grid rendering, neural network management
SimpleNeuralNetwork: core neural network logic (feedforward + backpropagation)
network_data.json: file where trained model is saved (auto-generated)
🔧 Possible Improvements

Add GUI (WinForms/WPF) or mouse input
Integrate real-world datasets (e.g. simplified MNIST)
Visualize learning progress or weight matrices
✍️ Author

Created with ❤️ by Manceriu Mădălin
