function OceanCleanupGA_Complete()
    % OCEANCLEANUPGA_COMPLETE
    % One big script that:
    % 1. Runs multiple GA trials of an LSTM-based ocean cleanup system.
    % 2. Plots average + std of best-fitness across runs.
    % 3. Visualizes environment (plastic distribution + currents).
    % 4. Saves best networks from each run.
    % 5. Finds the best-of-the-best and animates it.

    %% ==================== USER-PICKED PARAMETERS ====================
clc
clear all
close all
    % --- GA and Simulation Hyperparameters ---
    numRuns         = 10;   % how many separate GA trials
    populationSize  = 20;   % GA population
    numGenerations  = 50;   % how many generations per run
    crossoverRate   = 0.3;
    mutationRate    = 0.1;
    numTimesteps    = 50;   % length of each simulation
    elitismCount    = 2;    % how many elites survive each gen

    % --- LSTM Dimensions ---
    inputSize       = 5;   % [plasticDensity, currentX, currentY, agentX, agentY]
    hiddenSize      = 8;
    outputSize      = 2;   % [deltaX, deltaY]

    % If you want reproducibility, you can fix the seed or shuffle it:
    % rng('shuffle');  % comment/uncomment as needed

    %% ==================== STEP 1: CREATE ENVIRONMENT ONCE ====================
    env = createEnvironment();

    % (Optional) visualize environment right away
    figure('Name','Environment Overview','NumberTitle','off');
    visualizeEnvironment(env);

    %% ==================== STEP 2: MULTI-RUN GA PROCESS ====================
    % We'll store best-fitness-over-generations for each run:
    bestFitnessAllRuns = zeros(numGenerations, numRuns);

    % We'll store each run's final best network
    bestNetworks = cell(numRuns, 1);

    for runIdx = 1:numRuns
        fprintf('\n======== GA RUN %d of %d ========\n', runIdx, numRuns);
        % Each run: random initial population, evolve for N generations
        [bestFitnessOverGen, finalBestNetwork] = ...
            runSingleGA(env, populationSize, numGenerations, ...
                        crossoverRate, mutationRate, numTimesteps, ...
                        elitismCount, inputSize, hiddenSize, outputSize);

        bestFitnessAllRuns(:, runIdx) = bestFitnessOverGen;
        bestNetworks{runIdx} = finalBestNetwork;

        % Save the best network from this run to a file
        matFileName = sprintf('bestNetwork_run%d.mat', runIdx);
        save(matFileName, 'finalBestNetwork');
    end

    %% ==================== STEP 3: ANALYZE & PLOT MULTI-RUN STATS ====================
    meanFitness = mean(bestFitnessAllRuns, 2);   % average over runs (column-wise)
    stdFitness  = std(bestFitnessAllRuns, 0, 2); % std dev

    figure('Name','Multi-Run Fitness','NumberTitle','off');
    errorbar(1:numGenerations, meanFitness, stdFitness, 'LineWidth',2);
    xlabel('Generation');
    ylabel('Best Fitness (mean \pm std)');
    title(sprintf('GA Performance over %d Runs', numRuns));
    grid on;

    %% ==================== STEP 4: FIND THE BEST-OF-THE-BEST NETWORK ====================
    bestOfAll = -inf;
    championNetwork = [];
    for runIdx = 1:numRuns
        % Evaluate that run's final best again
        fitnessVal = simulateCleanup(bestNetworks{runIdx}, env, numTimesteps);
        if fitnessVal > bestOfAll
            bestOfAll = fitnessVal;
            championNetwork = bestNetworks{runIdx};
        end
    end
    fprintf('\n===== COMPLETED ALL RUNS =====\n');
    fprintf('Absolute best final network found has fitness = %.3f\n', bestOfAll);

    %% ==================== STEP 5: ANIMATE THE BEST-OF-THE-BEST ====================
    figure('Name','Champion Agent Animation','NumberTitle','off');
    animateAgent(championNetwork, env, numTimesteps);
end

%% ========================================================================
%% runSingleGA
% A helper function that runs the GA for a single run and returns:
% (1) bestFitnessOverGen [numGenerations x 1]
% (2) finalBestNetwork after evaluating the population post-generation
function [bestFitnessOverGen, finalBestNetwork] = ...
    runSingleGA(env, populationSize, numGenerations, crossoverRate, ...
                mutationRate, numTimesteps, elitismCount, ...
                inputSize, hiddenSize, outputSize)

    % 1) Initialize population
    population = cell(populationSize,1);
    for i = 1:populationSize
        population{i} = randomLSTM(inputSize, hiddenSize, outputSize);
    end

    % 2) Array to store best fitness per generation
    bestFitnessOverGen = zeros(numGenerations,1);

    % 3) GA Loop
    for gen = 1:numGenerations
        % Evaluate each individual's fitness
        fitnessVals = zeros(populationSize,1);
        for i = 1:populationSize
            fitnessVals(i) = simulateCleanup(population{i}, env, numTimesteps);
        end

        % Sort by fitness descending
        [fitnessVals, sortIdx] = sort(fitnessVals, 'descend');
        population = population(sortIdx);

        bestFitnessOverGen(gen) = fitnessVals(1);
        fprintf('Gen %d | Best Fitness = %.3f\n', gen, fitnessVals(1));

        % Selection & Next generation
        newPopulation = cell(populationSize,1);
        % Elitism
        for e = 1:elitismCount
            newPopulation{e} = population{e};
        end

        % Fill rest
        for k = (elitismCount+1):2:populationSize
            p1 = population{tournamentSelect(fitnessVals)};
            p2 = population{tournamentSelect(fitnessVals)};

            if rand < crossoverRate
                [c1, c2] = crossoverLSTM(p1, p2);
            else
                c1 = p1;
                c2 = p2;
            end

            if rand < mutationRate, c1 = mutateLSTM(c1); end
            if rand < mutationRate, c2 = mutateLSTM(c2); end

            newPopulation{k} = c1;
            if (k+1) <= populationSize
                newPopulation{k+1} = c2;
            end
        end
        population = newPopulation;
    end

    % Evaluate final population & pick best
    finalFitness = zeros(populationSize,1);
    for i = 1:populationSize
        finalFitness(i) = simulateCleanup(population{i}, env, numTimesteps);
    end
    [~, finalSortIdx] = sort(finalFitness, 'descend');
    finalBestNetwork = population{finalSortIdx(1)};
end

%% ========================================================================
%% createEnvironment
% For demonstration: 10x10 area with random plastic hotspots and sinusoidal currents
function env = createEnvironment()
    env.sizeX = 10;
    env.sizeY = 10;
    env.currentFieldX = @(x,y,t) 0.5 * sin(0.1*x + 0.05*t);
    env.currentFieldY = @(x,y,t) 0.5 * cos(0.1*y + 0.05*t);

    numHotspots = 5;
    env.plasticHotspots = rand(numHotspots, 3);
    env.plasticHotspots(:,1:2) = env.plasticHotspots(:,1:2)*10;
    env.plasticHotspots(:,3)   = env.plasticHotspots(:,3)*2 + 0.5;
end

%% ========================================================================
%% simulateCleanup
% Given an LSTM net, environment, and timesteps, returns total fitness
function fitness = simulateCleanup(lstmNet, env, numTimesteps)
    % Start agent in center
    agentPos     = [env.sizeX/2, env.sizeY/2];
    agentCapacity= 0;
    maxCapacity  = 100; % optional

    % LSTM hidden/cell states
    h_t = zeros(lstmNet.hiddenSize, 1);
    c_t = zeros(lstmNet.hiddenSize, 1);

    fitnessAccum = 0;

    for t = 1:numTimesteps
        plasticDensity = getPlasticDensity(agentPos, env);
        currentX = env.currentFieldX(agentPos(1), agentPos(2), t);
        currentY = env.currentFieldY(agentPos(1), agentPos(2), t);

        inputVec = [plasticDensity; currentX; currentY; agentPos(1); agentPos(2)];
        [dxdy, h_t, c_t] = lstmForward(inputVec, h_t, c_t, lstmNet);

        agentPos = agentPos + [dxdy(1)+currentX, dxdy(2)+currentY];
        agentPos(1) = max(0, min(env.sizeX, agentPos(1)));
        agentPos(2) = max(0, min(env.sizeY, agentPos(2)));

        % Collect plastic
        collectedNow = plasticDensity;
        if agentCapacity + collectedNow > maxCapacity
            collectedNow = maxCapacity - agentCapacity;
        end
        agentCapacity = agentCapacity + collectedNow;

        % Movement cost
        moveCost = norm(dxdy);
        incrementalFitness = collectedNow - 0.1*moveCost;
        fitnessAccum = fitnessAccum + incrementalFitness;
    end

    fitness = fitnessAccum;
end

%% ========================================================================
%% getPlasticDensity
% Simple radial hotspots
function density = getPlasticDensity(agentPos, env)
    x = agentPos(1);
    y = agentPos(2);
    sumDensity = 0;
    for i = 1:size(env.plasticHotspots,1)
        cx = env.plasticHotspots(i,1);
        cy = env.plasticHotspots(i,2);
        r  = env.plasticHotspots(i,3);

        dist = sqrt((x-cx)^2 + (y-cy)^2);
        if dist < r
            sumDensity = sumDensity + (1 - dist/r);
        end
    end
    density = sumDensity;
end

%% ========================================================================
%% lstmForward
% Single-step forward for the simple LSTM
function [output, h_next, c_next] = lstmForward(x_t, h_t, c_t, net)
    Wf = net.Wf; bf = net.bf;
    Wi = net.Wi; bi = net.bi;
    Wc = net.Wc; bc = net.bc;
    Wo = net.Wo; bo = net.bo;

    W_out = net.W_out;
    b_out = net.b_out;

    concat = [x_t; h_t];

    f_t = sigmoid(Wf*concat + bf);
    i_t = sigmoid(Wi*concat + bi);
    c_hat_t = tanh(Wc*concat + bc);
    c_next = f_t .* c_t + i_t .* c_hat_t;

    o_t = sigmoid(Wo*concat + bo);
    h_next = o_t .* tanh(c_next);

    output = W_out*h_next + b_out;
end

%% ========================================================================
%% randomLSTM
% Initialize random LSTM weights/biases
function net = randomLSTM(inputSize, hiddenSize, outputSize)
    limit = 0.1;
    net.inputSize  = inputSize;
    net.hiddenSize = hiddenSize;
    net.outputSize = outputSize;

    net.Wf   = randn(hiddenSize, inputSize+hiddenSize)*limit;
    net.Wi   = randn(hiddenSize, inputSize+hiddenSize)*limit;
    net.Wc   = randn(hiddenSize, inputSize+hiddenSize)*limit;
    net.Wo   = randn(hiddenSize, inputSize+hiddenSize)*limit;

    net.bf   = randn(hiddenSize,1)*limit;
    net.bi   = randn(hiddenSize,1)*limit;
    net.bc   = randn(hiddenSize,1)*limit;
    net.bo   = randn(hiddenSize,1)*limit;

    net.W_out= randn(outputSize, hiddenSize)*limit;
    net.b_out= randn(outputSize,1)*limit;
end

%% ========================================================================
%% crossoverLSTM
% Uniform crossover for all matrix fields
function [child1, child2] = crossoverLSTM(netA, netB)
    child1 = netA;
    child2 = netB;
    fields = {'Wf','Wi','Wc','Wo','bf','bi','bc','bo','W_out','b_out'};
    for f = 1:numel(fields)
        fname = fields{f};
        genesA = netA.(fname);
        genesB = netB.(fname);

        mask = rand(size(genesA))>0.5;
        newA = genesA;
        newB = genesB;

        newA(mask) = genesB(mask);
        newB(mask) = genesA(mask);

        child1.(fname) = newA;
        child2.(fname) = newB;
    end
end

%% ========================================================================
%% mutateLSTM
% Mutate ~10% of parameters with small Gaussian noise
function net = mutateLSTM(net)
    mutationStd = 0.02;
    fields = {'Wf','Wi','Wc','Wo','bf','bi','bc','bo','W_out','b_out'};
    for f = 1:numel(fields)
        fname = fields{f};
        genes = net.(fname);
        mask  = (rand(size(genes))<0.1);
        noise = mutationStd*randn(size(genes));
        genes(mask) = genes(mask) + noise(mask);
        net.(fname) = genes;
    end
end

%% ========================================================================
%% tournamentSelect
% Picks the better of two random indices from the population
function idx = tournamentSelect(fitnessVals)
    popSize = length(fitnessVals);
    c1 = randi(popSize);
    c2 = randi(popSize);
    if fitnessVals(c1)>fitnessVals(c2)
        idx = c1;
    else
        idx = c2;
    end
end

%% ========================================================================
%% visualizeEnvironment
% Plots a heatmap of plastic distribution + quiver for currents at t=0
function visualizeEnvironment(env)
    resolution = 20;
    xs = linspace(0, env.sizeX, resolution);
    ys = linspace(0, env.sizeY, resolution);
    [X, Y] = meshgrid(xs, ys);

    plasticMap = zeros(size(X));
    for i = 1:resolution
        for j = 1:resolution
            plasticMap(j,i) = getPlasticDensity([X(j,i), Y(j,i)], env);
        end
    end

    imagesc(xs, ys, plasticMap);
    axis xy;  % so y goes up
    hold on;
    colormap jet;
    colorbar;
    title('Plastic Distribution (Heatmap)');
    xlabel('X'); ylabel('Y');

    U = zeros(size(X));
    V = zeros(size(X));
    for i = 1:resolution
        for j = 1:resolution
            U(j,i) = env.currentFieldX(X(j,i), Y(j,i), 0);
            V(j,i) = env.currentFieldY(X(j,i), Y(j,i), 0);
        end
    end
    quiver(X, Y, U, V, 'k');
    hold off;
end

%% ========================================================================
%% animateAgent
% Show how a single network moves step by step
function animateAgent(lstmNet, env, numTimesteps)
    xlim([0 env.sizeX]); ylim([0 env.sizeY]);
    hold on; grid on;
    title('Champion Network Trajectory');
    xlabel('X'); ylabel('Y');

    agentPos = [env.sizeX/2, env.sizeY/2];
    agentPlot = plot(agentPos(1), agentPos(2), 'ro','MarkerFaceColor','r');

    h_t = zeros(lstmNet.hiddenSize,1);
    c_t = zeros(lstmNet.hiddenSize,1);

    pathX = zeros(numTimesteps,1);
    pathY = zeros(numTimesteps,1);

    for t = 1:numTimesteps
        plasticDensity = getPlasticDensity(agentPos, env);
        currentX = env.currentFieldX(agentPos(1), agentPos(2), t);
        currentY = env.currentFieldY(agentPos(1), agentPos(2), t);

        inputVec = [plasticDensity; currentX; currentY; agentPos(1); agentPos(2)];
        [dxdy, h_t, c_t] = lstmForward(inputVec, h_t, c_t, lstmNet);

        agentPos = agentPos + [dxdy(1)+currentX, dxdy(2)+currentY];
        agentPos(1) = max(0, min(env.sizeX, agentPos(1)));
        agentPos(2) = max(0, min(env.sizeY, agentPos(2)));

        pathX(t) = agentPos(1);
        pathY(t) = agentPos(2);

        set(agentPlot, 'XData', agentPos(1), 'YData', agentPos(2));
        drawnow;
        pause(0.05);
    end

    plot(pathX, pathY, 'r-');
    hold off;
end

%% ========================================================================
%% sigmoid
% Standard logistic
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end
