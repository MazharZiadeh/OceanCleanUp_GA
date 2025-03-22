🌊 Ocean Cleanup with LSTM + Genetic Algorithms


This project simulates an autonomous ocean cleanup agent powered by an LSTM neural network trained through Genetic Algorithms (GA). The goal: navigate dynamic ocean currents, collect plastic efficiently, and evolve smarter cleanup strategies over multiple generations.

🧠 What It Does
Evolves LSTM-based agents to collect ocean plastic.

Simulates plastic distribution + ocean currents in a 2D environment.

Runs multiple GA trials to optimize the agent's neural networks.

Saves and analyzes the best networks from each run.

Visualizes:

Environment (plastic & current fields)

Fitness stats (mean ± std across generations)

Best agent in action (animated path)

📦 Features
Modular GA Framework: Tournament selection, crossover, mutation, elitism.

Simple LSTM Implementation: Custom forward pass with full weight/bias control.

Dynamic Environment:

Sinusoidal ocean currents

Radial plastic hotspots

Fitness Function:

Rewards plastic collection

Penalizes excessive movement

🛠️ How to Run
matlab
Copy
Edit
clear all; close all;
OceanCleanupGA_Complete;
Requires MATLAB. No external toolboxes needed.

⚙️ Parameters (Editable in Script)
Parameter	Description	Default
numRuns	Number of GA trials	10
populationSize	Number of networks per GA run	20
numGenerations	How many generations to evolve	50
numTimesteps	Steps per agent simulation	50
inputSize	Input vector size for LSTM	5
hiddenSize	Hidden layer size of LSTM	8
outputSize	Output vector size (agent motion)	2
crossoverRate	Probability of crossover	0.3
mutationRate	Probability of mutation	0.1
elitismCount	Top individuals passed unchanged each gen	2
📊 Outputs
bestNetwork_runX.mat: Saved LSTM networks for each GA run.

Fitness plot: Mean ± standard deviation across runs.

Environment heatmap: Plastic density + current vectors.

Agent animation: Final champion network path in the environment.

🧬 How It Works (TL;DR)
Create Environment:

10x10 ocean with plastic hotspots & sinusoidal currents.

Run Genetic Algorithm:

Random LSTM weights.

Evaluate fitness via simulation.

Evolve over generations using crossover/mutation.

Track Best Networks:

Save best of each run.

Pick champion network across all runs.

Animate Best Agent:

Watch it clean plastic in real time.

🧠 Concepts Behind the Project
Neuroevolution: Training neural networks with evolutionary algorithms instead of backpropagation.

LSTM (Long Short-Term Memory): Memory-capable neural networks, enabling agents to learn temporal patterns.

Evolutionary Strategies: Genetic Algorithms used for search and optimization over parameter space.

Agent-based Modeling: Simulating intelligent behavior in dynamic, partially predictable environments.

📸 Screenshots
(Add GIFs or PNGs here if you have them!)

Fitness over generations

Environment heatmap

Best agent's animated trajectory

🔥 Why This is Cool
Self-evolving AI without gradients or labeled data.

Realistic environmental modeling.

Fully visual, interactive, and modifiable.

Foundation for real-world applications in autonomous marine robotics.

🧪 Potential Extensions
Use real ocean current data (e.g., NOAA datasets)

Multi-agent cooperation or competition

Upgrade LSTM with attention or GRU

Hardware deployment on underwater robots

Add 3D simulation and physics
