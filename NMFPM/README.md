"""Non-Negative Matrix Factorization - Profile Maker

Generate profiles from two non-negative matrices (X,C), whose product approximates the non-negative matrix Q of observed metals in quasar specttra. These profiles can be used to generate large libraries of realistic metal absorption profiles

 Parameters
----------
NMF_dct: Dictionary containing the information about the non-negative matrices (X,C).
    X : NMF_dct['X'] ndarray of shape 𝑛 × 𝑚, where 𝑚 is the number of reduced features in the NMF space
    C : NMF_dct['C'] ndarray of shape 𝑚 × 𝑣 and represents the coeffcient matrix of the 𝑚 reduced features

nsim: int, default = 1
    Number of profiles to be generated.

ion_family: {'moderate', 'low', 'user'}, default='moderate'.
    Ions families to be considered.
    Valid options:
    
    - 'moderate': If 'moderate' the profiles will follow a DeltaV_90 distribution typical of moderate ions transitions.
    
    - 'low': If 'low' the profiles will follow a DeltaV_90 distribution typical of low ions transitions
    
    - 'user': If 'user' the profiles will follow a DeltaV_90 distribution provided by the user with filename_ion_familiy
    
filename_ion_familiy: str, default=None
    User filename for DeltaV_90 pdf if ion_family = 'user'

ion_logN: ndarray of shape (nsim,), default=[14.0]
    log Ion Column Density in cm^-2

ion: str, default=['CIV']
    Ion transition to be simulated

trans_wl: float, default=[1548.2040]
    Ion transition wavelength in Angstrom
    
filename_ion_list: str, default=None
    User filename for lines' physical parameters

convolved: Boolean, default = False
    Allow for the generated profile to be convolved with a Gaussian kernel

res: float, default = 8
    Resolution of the generated profiles in km/s
    
px_scale = float, default = None
    Sampling of the generated profiles in km/s
    
SN = int, default = None
    Signal-to-Noise ratio of the continuum signal used to compute the gaussian noise to be added
    
sigma_sky = int, default = None
    RMS value of the sky signal. If not None the sky noise is computed as a random distribution centred on 0 and with dispersion sigma_sky
    
seed = int, default = None

verbosity = int, defalut = 0
    Print (1) or not (0) info to terminal


Attributes
----------

metals_ = Library of nsim synthetic metals
noise_ = Array of noise values for the nsim synthetic metals
wavelength_ = Array of wavelength values for the nsim synthetic metals


Authors
----------
A. Longobardi
"""

   Example
----------
import numpy as np
from nmf_profile_maker import NMFPM


nsim_=1000
ion_=['CIV'] *nsim_
trans_wl_ = [1548.2040]*nsim_
ion_logN_ = np.random.uniform(12,15,nsim_)

nmf_pm = NMFPM(nsim=nsim_,ion_family='moderate',ion_logN = ion_logN_, ion = ion_, trans_wl =trans_wl_,res=8)
simulated= nmf_pm.simulation() # Simulated Optical Depth profiles (not convolved nor rebinned)
metals = nmf_pm.metals_# Simulated Metal profile (convolved and rebinned if options are switched on)
Wavelength = nmf_pm.wavelength_# Array of wavelength values (rebinned if option is switched on)