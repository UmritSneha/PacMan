
## Comparing Memetic and Genetic Algorithm Applied To Pac-Man Game

Pac-Man is a popular real-time arcade game that provides a significantly rich and natural source of optimisation problem due to its innately stochastic nature. This paper describes an approach to developing and evaluating an artificial agent that replaces the human player to play an adapted version of the Pac-Man game. Five distinctive game layouts are chosen for optimisation such that each environment represents an increasing level of difficulty in achieving high scores through the different food dots and capsules positions. Contrary to previous works, a memetic algorithm is applied on each layout using a local search technique to reduce the likelihood of premature convergence.  In order to fairly evaluate the memetic agent, a genetic algorithm is implemented as a benchmark and is used for comparison. Both agents are tested by running several iterations. The experimental results on the various classes of problems show that the memetic agent outperforms the genetic agent only in complex environments while a simple game layout yield almost similar results from both algorithms. It can be further conjectured that both models perform well in less complex layouts while a simulation with an increased search space does not only take more computational time but also result in lower scores. 

### File Structure

**Implemented Files**

[`run.py`](run.py) Main file that runs the simulations. Contains both genetic and memetic algorithms as well as set up for game environment.

[`pacmanAgents.py`](pacmanAgents.py) Implementation of game elememts including ghosts, food dots and capsules using greedy search and shortest distance respectively.

[`plotResults.ipynb`](plotResults.ipynb) Visualising and comparing experimental results 

**Files from UC Berkeley**

[`pacman.py`](pacman.py) Describes pacman GameState type.

[`game.py`](game.py) Describes Agent and Direction.

[`graphicsDisplay.py`](graphicsDisplay.py)   Graphics for Pacman

[`graphicsUtils.py`](graphicsUtils.py)   Support for Pacman graphics

[`textDisplay.py`](textDisplay.py)   ASCII graphics for Pacman

[`ghostAgents.py`](ghostAgents.py)   Agents to control ghosts

[`keyboardAgents.py`](keyboardAgents.py)   Keyboard interfaces to control Pacman

[`layout.py`](layout.py)   Code for reading layout files and storing their contents

### Running experiments

```
python run.py
```

### Table of Results (Last Experiment Only)

![alt text](https://github.com/UmritSneha/PacMan/blob/main/plots/last_exp_table.PNG)

### Memetic Agent on Small Classic Layout (Random Trial)
![Alt Text](https://github.com/UmritSneha/PacMan/blob/main/plots/small_classic_ma_snippet.gif)

### Memetic Agent on Medium Classic Layout (Random Trial)
![Alt Text](https://github.com/UmritSneha/PacMan/blob/main/plots/medium_classic_ma_snippet.gif)


#### Environment Used
The Pac-Man simulation environment is modified and adapted from [Berkeley AI Pacman Search](https://github.com/jspacco/pac3man).

