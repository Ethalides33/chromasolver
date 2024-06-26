Here's a properly formatted version of your README file:

# Chromasolver

References for data files can be found in the manuscript.

## Input Data Format

The CSV file for input \( n \) and \( k \) values must have the following columns format:
```
material_wl_n, material_n, material_wl_k, material_k
```

## Running a Simulation

To run a typical simulation, you can use the following code:

```python
import numpy as np
import chromasolver as cs

# Define wavelength range for IR simulation
wlsIR = np.arange(2000, 20000, 1)

# Create a simulation object with the input data file
simIR = cs.Simulation(wlsIR, 'data_vo2_new_with_matchinglayers.csv')

# Define the stack configuration (material and thickness in nm)
stackPET = [
    ("Air", np.inf),
    ("PET", 1000),
    ("Air", np.inf)
]

# Run the simulation
resPET = simIR.sim_stack(stackPET, 1)
```

Ensure that you have the required CSV file (`data_vo2_new_with_matchinglayers.csv`) in the correct format before running the simulation.

---

Feel free to add any additional details or instructions as needed.
