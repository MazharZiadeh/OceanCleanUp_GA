https://www.researchgate.net/publication/390095494_A_Genetic_Algorithm-Driven_LSTM_Approach_for_Ocean_Plastic_Cleanup 

----------------------------------------------
OVERVIEW
----------------------------------------------
1. Multiple GA Trials
   - The script runs multiple independent GA trials (configurable via numRuns), each evolving a population of LSTM networks to maximize plastic collection while minimizing movement cost.

2. Generation-by-Generation Visualization
   - It saves and plots the best fitness scores across generations for each GA run, then displays aggregated statistics (mean ± standard deviation) across all runs.

3. Environment Simulation
   - Creates a randomized 2D grid with “plastic hotspots” and sinusoidal current fields. The agent uses an LSTM to decide movement actions ([deltaX, deltaY]).

4. Best Networks Saved
   - After each run, the best network is saved to a .mat file. The script identifies the overall champion network (best-of-the-best) and visualizes its performance via an animation.

5. Core Functions
   - simulateCleanup(...): Rolls out the LSTM agent in the environment and returns its fitness (total plastic collected minus movement penalty).
   - runSingleGA(...): Handles the GA loop for one run (initialization, selection, crossover, mutation, elitism).
   - visualizeEnvironment(...): Displays a heatmap of plastic distribution and a quiver plot of currents.
   - animateAgent(...): Animates the champion network’s path through the environment.
   - Utility: LSTM initialization and forward pass, crossover, mutation, and basic tournament selection.

----------------------------------------------
USAGE
----------------------------------------------
1. Clone/Download this repository.
2. In MATLAB (R2021 or later recommended), navigate to the folder containing OceanCleanupGA_Complete.m.
3. Run the script:

   clear all; close all;
   OceanCleanupGA_Complete;

4. Adjust Parameters inside OceanCleanupGA_Complete.m:
   - numRuns: Number of independent GA trials.
   - populationSize: GA population size.
   - numGenerations: GA evolution length per run.
   - numTimesteps: Simulation length for each agent evaluation.
   - inputSize, hiddenSize, outputSize: LSTM architecture dimensions.
   - crossoverRate, mutationRate, elitismCount: GA hyperparameters.

5. Results:
   - You’ll see plots of average best-fitness ± std dev over generations for all runs.
   - bestNetwork_runX.mat is saved for each run.
   - Finally, an animation of the champion (best overall) LSTM network’s movement is displayed.

----------------------------------------------
FILE STRUCTURE
----------------------------------------------
- OceanCleanupGA_Complete.m
  - Top-level Code
    - Parameters, environment creation, GA loop across multiple runs, final champion identification, and champion animation.
  - Helper Functions (within the same file)
    - runSingleGA()
    - simulateCleanup()
    - getPlasticDensity()
    - lstmForward()
    - randomLSTM()
    - crossoverLSTM()
    - mutateLSTM()
    - tournamentSelect()
    - visualizeEnvironment()
    - animateAgent()
    - sigmoid()

No additional toolboxes or dependencies are required. This script uses standard MATLAB functions.

----------------------------------------------
HOW IT WORKS
----------------------------------------------
1. Environment
   - A 10×10 area with randomly generated plastic hotspots. Currents are determined by sinusoidal functions of position and time.

2. LSTM Agent
   - Each LSTM agent receives local plastic density, current velocities, and its own [x, y] position as input, then outputs [Δx, Δy] actions.

3. Fitness Function
   - Maximizes total plastic collected minus a small cost for movement. The agent has an optional capacity limit to keep it realistic.

4. Genetic Algorithm Steps
   - Population Initialization: Each individual’s LSTM weights/biases are randomized.
   - Evaluation: Each individual is rolled out in the environment for numTimesteps; fitness is measured.
   - Selection: Tournament selection picks the better among random samples.
   - Crossover: Uniform crossover swaps partial genes (weights/biases) between parents.
   - Mutation: Some genes are perturbed with Gaussian noise.
   - Elitism: A few top individuals survive automatically to the next generation.

5. Best-of-the-Best
   - After all runs, each run’s final best network is re-evaluated to find the overall champion, which is then animated step by step in the environment.

----------------------------------------------
CONTRIBUTING
----------------------------------------------
Feel free to create pull requests or open issues to:
- Tweak parameters
- Model a more complex environment
- Explore multi-layer or alternate-activation LSTM architectures
- Incorporate additional performance metrics or new fitness functions

----------------------------------------------
