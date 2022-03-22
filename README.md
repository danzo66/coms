# mphys-research

The first program to run is interpolationfunction.py.
  This takes in the HD_163296_disk_structure.npz file (1).
  When the save() function is run, it:
  1) Calls interpolate2d for each of the 2d arrays
  2) Calls interpolate1d for each of the 1d arrays (this is not currently working, but as I do not actually need any 1D data, I have left it for now. This is why in abundancecalculation.py, several variables within the list data are just 0)
  3) Saves each interpolated array to an .npz file. This file is named based on the disc in question and the interpolation parameters
       
The second program to run is abundancecalculation.py.
  Set the variable args to be the molecule to calculate abundances for, the disc, and the interpolation parameters in the .npz file you want to load.
  When the write_to_file() function is called it:
  1) Calculates the molecular abundances
  2) Prints the desorption temperature and the snow line
  3) Writes this abundances and the disc structure out to two .txt files
    
The third program to run is a line survey code by Dr Catherine Walsh.
  Dr. Walsh is currently in the process of sharing this code.
  It is detailed in (2).
  It takes in the .txt files, and outputs three files:
    - (molecule_version)_profiles.out
    - (molecule_version)_intensities.out
    - (molecule_version)_continuum.out
    
The fourth and final program to run is spectraplotting.py.
  This takes in the .out files.
  When plot is run it:
  1) Plots the spectra
  2) Prints the disc integrated flux found in the line survey
  3) Prints the disc integrated flux from observations in (3)

(1) https://arxiv.org/abs/2109.06233
(2) https://arxiv.org/abs/1403.0390
(3) https://arxiv.org/abs/2109.06319
