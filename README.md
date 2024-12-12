# Advanced Multi-Agent Pathfinding Simulator with Machine Learning Heuristic

This repository contains the code and resources for an Advanced Multi-Agent Pathfinding Simulator that integrates traditional pathfinding algorithms with machine learning (ML) heuristics to enhance performance in dynamic environments.

## Project Overview

This project focuses on developing an Advanced Multi-Agent Pathfinding Simulator that not only implements a suite of traditional pathfinding algorithms but also integrates machine learning (ML) heuristics to enhance their performance in dynamic, multi-agent environments. By leveraging ML-based heuristics, the simulator seeks to improve the efficiency and adaptability of pathfinding algorithms, enabling them to better handle real-time changes and complex interactions among multiple agents. 

The simulator is designed with a graphical user interface (GUI) built using Tkinter, providing real-time visualization and interaction capabilities. Users can interactively place agents, obstacles, and weighted barriers on a grid-based platform, observe the resulting paths, and analyze the performance metrics of different algorithms. Additionally, dynamic obstacles that move autonomously add realism and complexity to the simulation, challenging agents to adapt their paths in real-time.

## Files and Their Purposes

* **Pathfinding_Simulator.py:** The main script to run the simulator GUI.
* **generate_dataset.py:**  Generates the dataset for training the machine learning model.
* **train_model.py:** Trains the machine learning model to predict heuristics.
* **heuristic_model.tflite:** The TensorFlow Lite model for real-time heuristic predictions.
* **heuristic_model_best.h5:** The best-performing Keras model during training.
* **heuristic_dataset.npy:** The dataset used to train the heuristic model.
* **scaler_mean.npy & scaler_scale.npy:** Store scaling parameters for feature normalization.
* **icon.ico:** An icon for the GUI window.
* **log_results.txt:** A text file to store the results of the simulation.

## How to Use the Simulator

1.  **Ensure you have the necessary dependencies:**
    *   Python 3.7 or higher
    *   Tkinter
    *   TensorFlow
    *   NumPy

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/FarhanSyed23/Pathfinding_Simulator.git
    ```

3.  **Run the simulator:**

    ```bash
    python Pathfinding_Simulator.py
    ```

4.  **Interact with the GUI:**
    *   Place agents on the grid by left-clicking.

<p align="center">
    <img width="800" src="https://github.com/FarhanSyed23/Pathfinding_Simulator/blob/main/Screenshots/Adding%20Weighted%20Obstacle.png" alt="Weighted Obstacles">
</p>

    *   Place Agents' Starting and End points and then Normal obstacles by right-clicking.

<p align="center">
    <img width="800" src="https://github.com/FarhanSyed23/Pathfinding_Simulator/blob/main/Screenshots/Adding%20Agents%20%26%20Normal%20Obstacles.png" alt="Agents & Normal Obstacles">
</p>

    *   Select algorithms and options from the dropdown menus.

<p align="center">
    <img width="800" src="https://github.com/FarhanSyed23/Pathfinding_Simulator/blob/main/Screenshots/Informed%20Algorithms.png" alt="Informed Algorithms">
</p>

<p align="center">
    <img width="800" src="https://github.com/FarhanSyed23/Pathfinding_Simulator/blob/main/Screenshots/Uninformed%20Algorithms.png" alt="Uninformed Algorithms">
</p>

    *   Click "Run All" to run the algorithm where Dynamic obstacles won't move and will just run on the present obstacles. 

<p align="center">
    <img width="800" src="https://github.com/FarhanSyed23/Pathfinding_Simulator/blob/main/Screenshots/Run%20All.png" alt="Run All">
</p>

    *   Start and stop the simulation using the buttons.

<p align="center">
    <img width="800" src="https://github.com/FarhanSyed23/Pathfinding_Simulator/blob/main/Screenshots/Start%20Simulation.png" alt="Start Simulation">
</p>

## Key Features

*   **Multiple pathfinding algorithms:**  Includes A\*, Dijkstra's, BFS, DFS, RRT, Jump Point Search, Theta\*, and more.
*   **Machine learning heuristic:**  Uses a trained neural network to improve pathfinding efficiency.
*   **Dynamic obstacles:**  Obstacles move around the grid, creating a more challenging environment.
*   **Collision avoidance:**  Agents can avoid colliding with each other.
*   **Real-time visualization:**  See the agents navigate the grid in real time.
*   **User-friendly interface:**  Easy to use and understand.

## Future Work

*   Enhancing collision avoidance in Theta\* and JPS.
*   Optimizing machine learning integration.
*   Expanding multi-agent support.
*   Integrating additional pathfinding algorithms.
*   Improving user experience.
