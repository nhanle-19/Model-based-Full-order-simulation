# Model-Based Full-Order Simulation

Introductory robotics-control simulations for linear inverted pendulum (LIP), hybrid LIP (H-LIP), and a Pinocchio-based five-link full-order walker.

## Contents

| File | Purpose |
| --- | --- |
| `LIP_main.m` | MATLAB/Simulink LIP simulation using `SIMrun_LIP.slx`. |
| `H_LIP_main_analytical.m` | MATLAB/Simulink H-LIP simulation using `SIMrun_H_LIP.slx`. |
| `H_LIP_main_ode45_solver.m` | MATLAB H-LIP solver using `ode45` instead of Simulink. |
| `animate_xz_com_foot.m` | Shared MATLAB animation helper for CoM and foot placement. |
| `Py_sim_H_LIP_solver.py` | Python H-LIP solver equivalent to the MATLAB `ode45` version. |
| `Beizer curve.py` | Small Bezier swing-foot trajectory demo. |
| `Full-Order-Fall.py` | Passive full-order five-link walker simulation. |
| `Full-Order_IO_Linearization_H_LIP.py` | Full-order IO linearization with H-LIP footstep adaptation. |
| `SIMrun_LIP.slx` | Simulink model for the LIP case. |
| `SIMrun_H_LIP.slx` | Simulink model for the H-LIP case. |

## Setup

### Python

Create an environment and install the shared Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The H-LIP and Bezier Python demos require only `numpy`, `scipy`, and `matplotlib`.

The full-order Python examples also require:

- `pin`, imported in code as `pinocchio`
- `five_link_walker.urdf` placed in this repo directory
- any visual mesh files referenced by that URDF

The URDF is not currently included in this repository, so the full-order scripts will stop with a clear missing-file message until it is added.

Install the extra Python package for those examples with:

```bash
pip install -r requirements-full-order.txt
```

### MATLAB

Run MATLAB from this repository directory so that scripts can find the `.slx` files and `animate_xz_com_foot.m`.

## Running

MATLAB:

```matlab
LIP_main
H_LIP_main_analytical
H_LIP_main_ode45_solver
```

Python:

```bash
python3 Py_sim_H_LIP_solver.py
python3 "Beizer curve.py"
```

For a terminal-only Python H-LIP check:

```bash
python3 Py_sim_H_LIP_solver.py --no-plot
```

Full-order examples, after adding the URDF and installing Pinocchio:

```bash
python3 Full-Order-Fall.py
python3 Full-Order_IO_Linearization_H_LIP.py
```

## Model Notes

- LIP uses constant CoM height and single-support pendulum dynamics.
- H-LIP adds a discrete foot-placement update across single-support and double-support phases.
- The full-order examples use a five-link planar walker and add point frames at the left and right feet for stance/swing tracking.
- `Full-Order_IO_Linearization_H_LIP.py` combines swing-foot Bezier tracking, stance-foot constraints, CoM tracking, and H-LIP deadbeat step-length adaptation.

## Quick Verification

The Python files can be syntax-checked with:

```bash
python3 -m compileall -q .
```

The no-plot H-LIP run should print CoM and foot x-position arrays:

```bash
python3 Py_sim_H_LIP_solver.py --no-plot
```
