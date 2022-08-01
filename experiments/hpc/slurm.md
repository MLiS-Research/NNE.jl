# How to run code on a cluster

This code only supports SLURM. 

First of all, create a batch script as you normally would:
```bash
#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1024
#SBATCH --time=00:30:00
#SBATCH -o hpc/output/test_job_%j.out
```
Then load all the environment variables (all the `module load` commands). For example, if they are in a bash script:
```bash
source ~/juliaenv.sh
```

Then finally include the run command:
```bash
wd=$(pwd)
code="[println(remotecall_fetch(print_resources, w)) for w in workers()]"
julia --project hpc/setup_processes.jl --working_dir $wd --include_file "hpc/test_workers.jl" -e "$code"
``` 

This is all you need to get started.