# Reinforcement Learning Maze Solver: SARSA vs. Expected SARSA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Visualization-green)
![NumPy](https://img.shields.io/badge/NumPy-Computation-orange)

##  Project Overview

This project provides a dynamic, visual comparison of two classic Reinforcement Learning (RL) algorithms: **SARSA (State-Action-Reward-State-Action)** and **Expected SARSA**.

The core of the project is a custom-built, procedurally generated grid-world environment. Agents are trained to navigate from a starting quadrant to a goal in the opposite quadrant while avoiding procedurally placed obstacles. The project emphasizes the visualization of the learning process, offering both real-time training animations via OpenCV and final static path comparisons to analyze how different algorithms converge on a solution.

##  Key Features

* **Dual Algorithm Implementation:** Direct comparison of On-Policy SARSA vs. Expected SARSA learning behaviors.
* **Procedural Environment:** A highly modular grid-world where grid size and wall density are configurable, ensuring no two training runs are exactly the same.
* **Sophisticated Reward Shaping:**
    * **Goal:** High positive reward (+).
    * **Walls/Obstacles:** Large negative penalty (-).
    * **Heuristics:** Small proximity "nudges" (positive/negative) based on distance to the objective to guide exploration.
* **OpenCV Visualization:**
    * **Live Training:** Watch the agent explore (Blue Block) and learn in real-time.
    * **Static Comparison:** Automatically generates a side-by-side image of the optimal paths learned by both agents.

## Tech Stack

* **Language:** Python
* **Libraries:**
    * `numpy`: Matrix operations, Q-Table management, and grid logic.
    * `opencv-python`: Real-time rendering and final image generation.
    * `random`: Procedural generation logic for obstacles and start/end points.

---

## How It Works

### The Environment
The agent operates in a flexible grid (e.g., 10x10 or 20x20). To maximize path complexity, start and end points are generated in opposite corners. Walls are generated randomly based on a density parameter.

### Algorithm Logic

Both agents utilize a **Q-Table** to store state-action values, but they differ in how they update those values during training.

#### 1. SARSA (On-Policy)
SARSA updates the Q-value based on the action *actually taken* by the current policy. The update rule is:

$$Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]$$

Where $S'$ is the next state and $A'$ is the next action picked by the policy.

#### 2. Expected SARSA
Expected SARSA updates the Q-value based on the *expected value* of the next state, averaging over all possible actions weighted by their probability. This usually results in lower variance updates.

$$Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \sum_{a} \pi(a|S') Q(S', a) - Q(S, A)]$$

### Visual Legend
The visualization window uses the following color scheme:
* **🟦 Blue Block:** The Agent
* **🟩 Green Block:** The Goal
* **⬛ Black Blocks:** Walls/Obstacles
* **🔴 Red/Green Lines:** The optimal path taken during final evaluation.

---

##  Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/dagaaryan011/Reinforcents.git](https://github.com/dagaaryan011/Reinforcents.git)
cd Reinforcents
```

### 2. Install dependencies
Ensure you have Python installed, then run:
```bash
pip install numpy opencv-python
```

### 3. Run the simulation
To start the training process and visualization:

```bash
python Expected_sarsa.py
```
*(Note: If your main script has a different name, replace `Expected_sarsa.py` with your filename).*

##  Results

Upon completion of the training episodes, the program will:
1.  Save the Q-Tables.
2.  Display a **Comparison Image**.

This image visualizes the path taken by the SARSA agent versus the path taken by the Expected SARSA agent, allowing for immediate visual analysis of which algorithm found a safer or shorter path.

##  Contributing

Contributions are welcome! If you have ideas for new environments, different algorithms (like Q-Learning), or better visualization features:

1.  Fork the Project
2.  Create your  Branch 
3.  Commit your Changes (`git commit -m 'commit msg'`)
4.  Push to the Branch (`git push origin branch_name`)
5.  Open a Pull Request
