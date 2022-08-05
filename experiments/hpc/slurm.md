# How to run code on a cluster

This code only supports SLURM. Make sure you are current in the `experiments` directory:
```bash
cd experiments
```

First of all, create a batch script as you normally would:
```bash
#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=4
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=50G
#SBATCH -o hpc/output/mnist_%j.out
```
Then load all the environment variables (all the `module load` commands). For example, if they are in a bash script:
```bash
source ~/juliaenv.sh
```

Make sure the directory `hpc/output` is created before running, or SLURM will silently fail:
```bash
mkdir hpc/output
```

Then finally include the run command:
```bash
wd=$(pwd)
run_file="$wd/scripts/mnist_ex_1.jl"
julia --project hpc/setup_processes.jl --working_dir $wd --run_file "$run_file"
``` 

This is all you need to get started.