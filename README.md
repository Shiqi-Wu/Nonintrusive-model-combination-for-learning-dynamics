# Nonintrusive-model-combination-for-learning-dynamics
# Experiment Code for Manuscript Preparation

This repository contains the code and related materials for the experiments conducted in preparation for my manuscript: "Non-intrusive model combination for learning dynamics". The code and data provided here are intended to support and reproduce the results presented in the manuscript. This README file provides an overview of the repository's structure and how to use the code and data.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Introduction

In data-driven modelling of complex dynamic processes, it is often desirable to combine different classes of models to enhance performance. Examples include coupled models of different fidelities, or hybrid models based on physical knowledge and data-driven strategies. A key limitation of the broad adoption of model combination in applications is intrusiveness: training combined models typically requires significant modifications to the learning algorithm implementations, which may often be already well-developed and optimized for individual model spaces. In this work, we propose an iterative, non-intrusive methodology to combine two model spaces to learn dynamics from data. We show that this can be understood, at least in the linear setting, as finding the optimal solution in the direct sum of the two hypothesis spaces, while leveraging only the projection operators in each individual space. Hence, the proposed algorithm can be viewed as iterative projections, for which we can obtain estimates on its convergence properties. This repository contains code and data used for experiments conducted as part of manuscript preparation. 

## Repository Structure

The repository is organized as follows:

```
/
|-- toy-model/                 # Source code for running the reaction-diffsuion equation example (section 4.1) and convergence example in section 2
|-- cardiac-coupled-system/                 # Source code for running experiments in section 4.2
|-- robotics-control/              # Source code for running experiments in section 4.3
|-- utils/              # Code for plotting
|-- manuscript/           # Manuscript document (not included)
|-- README.md             # This README file
```

It looks like you have a clear structure in your repository, but you can improve the explanation for the files in your "toy-model" directory. Here's a revised version:

- **toy-model:** This directory `toy-mode/` contains the source code and Jupyter Notebook files for two listed experiments:
  - `Toy_model.py` is the source code for training the model and generating data.
  - `reaction-diffusion-equation.ipynb` is used to execute and document the experiments presented in section 4.1 and to showcase the obtained results.
  - `subspace-angle.ipynb` is employed to conduct the experiments described in section 3 and to present the corresponding results.

- **cardiac-coupled-system:** The `cardiac-coupled-system/` directory contains the source code used to conduct the experiments in section 4.2, the cardiac PDE-ODE system. 
  - `iteratively-train.py` is the source code for training the model using Algorithm 1.
  - `acceleration-train.py` is the source code for training the model using Algorithm 2.
  - `generate-data.py` is the source code for generating the training or testing data.
  - `residual-learning.py` is the source code for constructing the residual learning model.
  - `KoopmanDL.py` contains the source code of parameterized Koopman model.
  - `Hybrid.py` contains the source code of Hybrid models.
  - `Cardiac_Electrophysiology.py` contains the details of the PDE-ODE coupled system.
  - `plot.ipynb` could be used to evaluate the models and present the results.

- **robotics-control:** This directory `robotics-control/` contains the codes for experments in section 4.1: the robotics fish example.
  - `train.ipynb` is the Jupyer Notebook file to generate the training data and train the hybrid structures and nonlinear structure.
  - `generate_test_data.py` is the code for generating the test data in the tracking problem.
  - `linear_predictor.py`, `hybrid_predictor_1.py`, `hybrid_predictor_1.py`, `bilinear_predictor.py` are the codes for the tracking problem respectively with the linear structure, hybrid structure 1, hybrid structure 2, nonlinear structure 3. One should run the file as 
  ```bash
  python linear_predictor.py 0
  ```
  with the parameter as [0: $\omega_a = 2\pi$; 1: $\omega_a = 2\pi + \pi \sin(\frac{\pi}{60}t)$; 2: random $\omega_a$].
  - `MPCsolver.py` contains the code for MPC optimization problem.
  - `parameters.py` contains the parameters for generating data and model.
  - `robotics_dym.py` contains the detailed information of the system.
  - `plot.ipynb` is the Jupyter Notebook file to present the results.

- **manuscript:** While the manuscript itself is not included in this repository, you can refer to the manuscript for detailed information about the research, methodology, and results. The manuscript should provide context and reference to this code and data.

## Requirements

Before using the code in this repository, ensure that you have the following software and libraries installed:

- Python 3.8.16
- tensorflow                    2.11.0
- tensorflow-gpu                2.11.0
- numpy                         1.24.2
- scipy                         1.10.1
- seaborn                       0.12.2
- matplotlib                    3.6.3
- scikit-learn                  1.2.1

## Usage

To use this repository for replicating the experiments:

Clone this repository to your local machine.

   ```bash
   git clone https://github.com/Shiqi-Wu/Nonintrusive-model-combination-for-learning-dynamics
   ```

Please note that you should have the necessary dependencies and data in place to run the code successfully.

## License

MIT License

Copyright (c) [2023] [Shiqi Wu]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Acknowledgments
We extend our appreciation to Yue Guo from the National University of Singapore for generously sharing the code for the Parameterized Koopman model, which greatly contributed to our research and experimentation.