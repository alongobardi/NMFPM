   """Non-Negative Matrix Factorization - Profile Maker
    
    Generate profiles from two non-negative matrices (X,C), whose product approximates the non-negative matrix Q of observed metals in quasar spectra. These profiles can be used to generate large libraries of realistic metal absorption profiles
    
    Parameters
    ----------
    NMF_dct: Dictionary containing the information about the non-negative matrices (X,C).
        X : NMF_dct['X'] ndarray of shape n x m, where m is the number of reduced features in the NMF space
        C : NMF_dct['C'] ndarray of shape m x u and represents the coeffcient matrix of the m reduced features
    
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
        Resolution of the generated profiles in km/s. Needs to set convolved True to take effect
        
    px_scale = float, default = None
        Sampling of the generated profiles in km/s
        
    SN = ndarray of shape (nsim,), default = [None]
        Signal-to-Noise ratio of the continuum signal used to compute the gaussian noise to be added
        
    sigma_sky = int, default = None
        RMS value of the sky signal. If not None the sky noise is computed as a random distribution centred on 0 
        and with dispersion sigma_sky
    
    doublet: Boolean, default = False
        Enables creation of doublets (e.g. MgII, CIV etc...)
    
    dbl_fratio: float, default = 0
        If doublet True, create a second line with oscillator strengh 
        f_line_2 = dbl_fratio * f_line_1


    dbl_dvel: float, default = 0
        If doublet True, create a second line with center shifted in velocity by dbl_dvel [km/s]

    seed: int, default = None
        Allow selection of seed
        
    verbosity = int, defalut = 0
        Print (1) or not (0) info to terminal
    
    
    Attributes
    ----------

    flux = nsim synthetic spectra, with noise if so desired
    flux_nonoise = nsim synthetic spectra, no noise
    noise = associated noise values 
    wave =  wavelength values for the nsim synthetic spectra
    ew   =  E.W. (A) distribution of the profiles (intergates on doublet if present)
    wave_native = original wave before resampling/convolution
    flux_native = original flux before resampling/convolution

    
    Authors
    ----------
    A. Longobardi
    """
    
    Examples
    ----------
    
    import numpy as np
    from nmfpm.nmf_profile_maker import NMFPM


   
    - Example 1: Generate 10^3 CIV1548.204 absorbers (no doublet), at infinite S/N, 
	         with a resolution of 8 km/s and a pixel sampling of 1 km/s
      
    

    nsim_=1000
    ion_=['CIV'] *nsim_
    trans_wl_ = [1548.2040]*nsim_
    ion_logN_ = np.random.uniform(12,15,nsim_)

    nmf_pm_spc = NMFPM(nsim=nsim_,ion_family='moderate',ion_logN = ion_logN_, ion = ion_, trans_wl =trans_wl_)
    simulated = nmf_pm_spc.simulation() # nsim simulated Optical Depth profiles (not convolved nor rebinned)
    fluxes = nmf_pm_spc.flux  # nsim fluxes 

    - Example 2: Generate 10^5 MgII absorbers with doublet, at S/N varying in range 2.5 <= S/N <= 15
	         at a resolution of 60 km/s and a pixel sampling of 16 km/s.
	         Anable to print NMF-PM additional logs.


    nsim_=1000000
    ion_=['MgII']*nsim_
    trans_wl_ = [2796.3543]*nsim_
    ion_logN_ = np.random.uniform(13,15.5,nsim_)
    SN_= np.random.uniform(2.5,15,nsim_)
    px_scale=16
    res=60
    doublet=True
    dbl_fratio = [2.01]*nsim
    dbl_dvel = [770]*nsim

    nmf_pm_spc = NMFPM(nsim=nsim_,ion_family='low',ion_logN=ion_logN_, ion=ion_, trans_wl = trans_wl_,\
                       res=res,convolved=True,px_scale=px_scale,SN=SN_\,
                       doublet=doublet,dbl_fratio=dbl_fratio_,dbl_dvel=dbl_dvel_,verbosity=1)

    simulated = nmf_pm_spc.simulation() # nsim simulated optical Depth profiles (not convolved nor rebinned)
    fluxes = nmf_pm_spc.flux  # nsim fluxes (with noise, convolved and rebinned as required)
   
