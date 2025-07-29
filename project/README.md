*Setup for running the code on GPU*
#####################################################

For the python version: 

1-allocat a GPU card: salloc -C gpu -q interactive -t 30 --gpus=1 --gpu-bind=none
2-install all the modules required for the pipeline (CUDA modules and install CuPy) in a virtual environment or use conda
module load CUDA 
pip install cupy-cuda12x   # Replace x with your CUDA version (e.g. 12.1 -> cupy-cuda121)
3- use an env that can run cosmosis 
4- use the command "cosmosis boost_factor.ini" to run the code on a GPU  
5-to run the job: sbatch bf_job.sh
6-Monitor the queue: squeue -u $USER

#####################################################
For the C++ version: 
1- run make file to builed the code.
2- Update your .ini file: uncommand file = /path/to/boost_factor_likelihood.so
3- run cosmosis: cosmosis boost_factor.ini


to run the CUDA version: 

1- make
2- cosmosis boost_factor.ini
3- try job.sh 
