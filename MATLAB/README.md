# MATLAB
This folder contains the MATLAB implementation of `Kernel-Embedding-for-PGibbs` which uses [CasADi](https://web.casadi.org/), [IPOPT](https://coin-or.github.io/Ipopt/), and the proprietary [HSL Linear Solvers](https://licences.stfc.ac.uk/product/coin-hsl) to solve the scenario optimal control problem.

**Note that the results presented in the paper were obtained using the Julia version. Although the MATLAB version provides similar results, it does not reproduce these results exactly due to different random numbers in Matlab**

To execute the MATLAB code, first install [MATLAB](https://mathworks.com/products/matlab.html). 

Then, install the proprietary [HSL Linear Solvers](https://licences.stfc.ac.uk/product/coin-hsl). You can find detailed instructions for the installation [here](HSL_install_instructions.md). Executing the code without the HSL Linear Solvers may be possible, but this is not recommended due to the long runtime. In this case, change the *'linear_solver'* option in the struct solver_opts, e.g., to *"mumps"*.

Finally, download [CasADi](https://web.casadi.org/get/) and unzip the source code. Afterward, add the CasADi directory to the MATLAB path by editing the command
```
addpath('<yourpath>/casadi-3.6.5-windows64-matlab2018b')
```
at the beginning of the MATLAB scripts `PG_OCP_Kernel.m`, `PG_OCP_Kernel_nonlinear.m`, or `PG_OCP_Kernel_CorridorTest.m` in the folder `examples`.

Then, execute the scripts `PG_OCP_Kernel.m`, `PG_OCP_Kernel_nonlinear.m`, or `PG_OCP_Kernel_CorridorTest.m` in the folder `examples`.

Tested with Windows 11 and MATLAB R2024a.
