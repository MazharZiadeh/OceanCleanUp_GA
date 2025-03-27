https://www.researchgate.net/publication/390095494_A_Genetic_Algorithm-Driven_LSTM_Approach_for_Ocean_Plastic_Cleanup 

**README: Ocean Plastic Cleanup with LSTM + Genetic Algorithm**

---

## Overview
This repository demonstrates a *Genetic Algorithm (GA)* approach to training an *LSTM-based agent* for autonomous ocean plastic collection. The goal is to develop a policy that navigates a simulated environment, gathers plastic debris efficiently, and balances movement costs against the amount of plastic collected.

---

## Key Components
1. **Environment**  
   - A 2D ocean grid (`env.sizeX`, `env.sizeY`) with *plastic hotspots* and *dynamic ocean currents*.
   - The agent starts near the center and moves around collecting plastic.

2. **LSTM Agent**  
   - Takes a 5D input: `[plasticDensity, currentX, currentY, agentX, agentY]`.
   - Outputs `[deltaX, deltaY]` movement commands.
   - Evolved via a GA by mutating and recombining the LSTM’s weights.

3. **Genetic Algorithm**  
   - *Population-based search* (e.g., 30–50 individuals).
   - *Crossover & Mutation* operators to evolve LSTM weights.
   - *Fitness* = total plastic collected – (movement penalty).

4. **Multiple GA Runs**  
   - We run ~50 independent trials, each with up to 200 generations (or more).  
   - Each run logs `Gen # | Best Fitness = <value>` to track improvement over time.

---

## Notable Results
After tuning hyperparameters (population size, mutation rate, generations, etc.), some GA runs began reaching **fitness values in the 30–40+ range** by the 200th generation. Below is a summary of key findings:

- **Early Generations:** Fitness often starts in the single digits or teens, as the initial random policies barely collect plastic.
- **Mid Generations (50–150):** Most runs show steady gains. Some plateau while others leap ahead if the population stumbles upon a more efficient trajectory.
- **Late Generations (150–200+):**  
  - Several runs exceed 30 fitness, with some surpassing 40 and even hitting ~43 in the best cases.  
  - The agent refines its strategy to remain near dense hotspots and keep movement cost manageable.

Overall, the improvements confirm that:
- **Longer GA runs** or **better hyperparameters** (e.g., bigger population, higher/lower mutation, refined crossover) significantly boost final fitness.
- **Fitness** can vary widely across runs due to stochastic initialization and random seeds, but on average we see a substantial improvement compared to shorter or poorly tuned runs.

---

## Usage

1. **Run the Main Script**  
   - The primary entry point is `OceanCleanupGA_Complete.m`.  
   - Adjust hyperparameters at the top to match your system or experimentation needs (e.g., `numRuns`, `numGenerations`, `populationSize`).
   - Run in MATLAB/Octave:
     ```matlab
     OceanCleanupGA_Complete
     ```
2. **Analyze Results**  
   - During the GA, you’ll see console logs like `Gen # | Best Fitness = …` for each run.  
   - Post-run, check out:
     - *Mean and standard deviation* plots of best fitness (across multiple runs).
     - The final “champion” agent animation showing how it navigates the ocean.
![image](https://github.com/user-attachments/assets/cc0d5d95-7075-44c0-a756-4ada581ccc80)
![image](https://github.com/user-attachments/assets/aa61efcb-14a2-4ef2-be29-d4d5482106b9)
![image](https://github.com/user-attachments/assets/a8aeba9b-d88c-4712-8e81-af9abc92b666)



---

## Tips & Tweaks
- **Increase Generations**: If your runs plateau early, give them more time (200+ generations).  
- **Adjust Mutation/Crossover**: If you need more exploration, raise mutation (0.15+). If solutions converge too slowly, lower mutation or increase elitism.  
- **Population Size**: Larger populations (50+) can produce more diverse solutions and avoid local optima, but require more computation.

---

## Contributing
Feel free to open issues or pull requests if you spot improvements, optimizations, or wish to add features like a more realistic ocean current model or advanced LSTM architectures.

---
