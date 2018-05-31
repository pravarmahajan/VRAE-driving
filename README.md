## Variational Recurrent Autoencoder for Driving Risk Prediction

This code has been taken from [Variational-Autoencoder](http://arxiv.org/pdf/1412.6581.pdf) implementation [by y0ast](https://github.com/y0ast/Variational-Recurrent-Autoencoder)

## Installing Dependencies

### Before using Conda (Everytime)
Since `conda` is available with python3, you need to load python3 module to be able to use conda:
```
module load python/3.5
```

### Creating new Conda Environment (First Time)
This will create a new environment named `menv` and install all the dependencies

```
conda env create -f environment.yml
```

### Loading Environment (Everytime)
```
source activate myenv
```
Remember to load python 3.5 before loading the environment

## Data
The code assumes that all your data is in `data` directory. 
```
~$ ls data/
smallSample_10_10           smallSample_10_200_keys.pkl  smallSample_10_50.npy        smallSample_5_20           smallSample_5_5_keys.pkl
smallSample_10_10_keys.pkl  smallSample_10_200.npy       smallSample_50_200           smallSample_5_20_keys.pkl  smallSample_5_5.npy
smallSample_10_10.npy       smallSample_10_50            smallSample_50_200_keys.pkl  smallSample_5_20.npy
smallSample_10_200          smallSample_10_50_keys.pkl   smallSample_50_200.npy       smallSample_5_5
```
The risk data is assumed to be available at
`/users/PAS0536/osu9965/Telematics/Trajectories/Selected Drivers/DriverIdToRisk.csv`

## Running the script

To run for testing on a small dataset, use:
```
sh run.sh
```

A submission script is available for submitting it on cluster:
```
qsub run.sub
```

The list of options available for passing as command line arguments is given in `run.py`script, `parse_args` function
