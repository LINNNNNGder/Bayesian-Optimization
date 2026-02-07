# Batch Multi-Objective Bayesian Optimization Engine
## 1. Overview
Paper-MOBO is a lightweight, reproducible Batch Multi-Objective Bayesian Optimization (MOBO) engine designed to automatically explore design spaces using a user-defined objective_function().The system is built for single-machine multi-GPU, long-running computational workloads, and research-grade reproducibility, with automatic resume and batch parallel optimization.

Typical use cases:
- Materials / molecular simulation parameter optimization
- Machine learning hyperparameter optimization
- Black-box physical or engineering optimization
- Multi-objective design space exploration

## 2. Key Features
Supports arbitrary-dimension design variables

Supports any number of objectives (multi-objective optimization)

Batch Bayesian optimization using qEHVI

Single-machine multi-GPU parallel execution

Generic objective_function() interface

Automatic resume after interruption

CSV database records all optimization history

Safe for long-running tasks (hour-level simulations)

Fully configuration-driven (no source modification required)

## 3. Project Structure
```
paper-mobo/
│
├── configs/
│   └── exp.yaml              # Experiment configuration
│
├── src/
│   ├── mobo.py               # BO core (qEHVI)
│   ├── runner.py             # Multi-GPU batch execution
│   ├── pipeline.py           # Main optimization loop
│   ├── config.py             # Config loader
│   └── schema.py             # Variable/objective parser
│
├── user/
│   └── objective.py          # User-defined objective function
│
├── data/
│   └── input_output.csv      # Optimization database (auto-created)
│
└── README.md
```
## 4. Quick Start
### 4.1 Install Dependencies
Requires:
```
PyTorch
BoTorch
gpytorch
multiprocessing
```
### 4.2 Define the Objective Function
Edit:
`user/objective.py`

Example:

```
def objective_function(x: dict, gpu_id=None, device=None) -> dict:
	"""
	Input: design variables
	Output: objective values (dict)
	"""

	# Example (replace with real simulation)
	f1 = -(x["x1"] - 0.3)**2
	f2 = -(x["x2"] - 0.7)**2

	return {
		"obj1": f1,
		"obj2": f2,
	}
```

### 4.3 Run Optimization
python -m src.pipeline --config configs/exp.yaml

The system will:
-Initialize database
-Generate initial samples
-Run Bayesian optimization loop
-Execute objective function in parallel on multiple GPUs
-Store all results in database
-Support automatic resume

### 4.4 Resume After Interruption

Simply rerun:`python -m src.pipeline --config configs/exp.yaml`.The system automatically resumes unfinished jobs.

## 5. Configuration

Configuration file:`configs/exp.yaml`

### 5.1 Design Variables
Defines optimization variables and bounds.
design_variables:
-names: [x1, x2]
bounds:
-lower: [0, 0]
-upper: [1, 1]

### 5.2 Objectives
objectives:
-names: [obj1, obj2]
Must match keys returned by objective_function().

### 5.3 Bayesian Optimization Parameters
```
bo:
  q_batch: 2
  num_initial_samples: 6
  max_iterations: 100
  ref_margin: 0.05
  num_restarts: 10
  raw_samples: 128
  mc_samples: 128

### 5.4 GPU Configuration
hardware:
  gpu_ids: [0, 1]
  mp_start_method: spawn

### 5.5 Objective Function Entry
objective:
  callable: "user.objective:objective_function"
  parallel: true
  timeout_seconds: 0
  max_retries: 1

## 6. Optimization Workflow

The optimization process proceeds as follows:

Initialize database
Automatically creates data/input_output.csv

Initial sampling
Uses LHS or random sampling

Fit multi-objective GP model

Batch BO proposal (qEHVI)
Proposes new candidate designs

Multi-GPU parallel execution
Runs objective function on assigned GPUs

Database update
Stores results, runtime, and status

Resume capability
Unfinished samples automatically continue
