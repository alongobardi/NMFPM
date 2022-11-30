import os
import numbers
import numpy as np
import pickle
import time
import pandas as pd
import random
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from collections import defaultdict
from bisect import bisect_left
from astropy import units as u



class NMFPM(object):
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
        RMS value of the sky signal. If not None the sky noise is computed as a random distribution centred on 0 and with dispersion sigma_sky
    
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

    flux = Nsim synthetic spectra, with noise if so desired
    flux_nonoise = Nsim synthetic spectra, no noise
    noise = associated noise values 
    wave =  wavelength values for the nsim synthetic spectra
    
    
    Authors
    ----------
    A. Longobardi
    """

    def __init__(
        self,
        nsim = 1,
        ion_family = 'moderate',
        filename_ion_family = None,
        ion_logN = [14.0],
        ion = ['CIV'],
        trans_wl = [1548.2040],
        filename_ion_list = None,
        convolved = False,
        res = 8,
        px_scale = None,
        SN = [None],
        sigma_sky = None,
        doublet = False, 
        dbl_fratio = 0.0,
        dbl_dvel = 0.0,
        seed = None,
        verbosity= 0
        
    
    
    ):
        
        self.nsim = nsim
        self.ion_family = ion_family
        self.filename_ion_family = filename_ion_family
        self.ion_logN = np.array(ion_logN)
        self.ion = np.array(ion)
        self.trans_wl = np.array(trans_wl)
        self.filename_ion_list = filename_ion_list
        self.convolved = convolved
        self.res = res
        self.px_scale = px_scale
        self.SN = np.array(SN)
        self.sigma_sky = sigma_sky
        self.doublet = doublet 
        self.dbl_fratio = dbl_fratio
        self.dbl_dvel = dbl_dvel
        self.seed = seed
        self.verbosity=verbosity
        
        #define index array 
        self.index=np.arange(self.nsim)
      
        # Run saftey checks
        # Check parameters
        self._check_params()
        # Check files
        self._check_files()
        
        if self.verbosity > 0:
            print("NMF-PM: Starting preliminary operations")

        if self.verbosity > 0:
            print("NMF-PM: Load input velocity profiles and oscillator strengths")

        # Load up some useful files
        __path__ = os.path.dirname(os.path.realpath(__file__))
        
        with open( __path__+'/''docs/NMF_dictionary.json', "rb") as fp:
            self.NMF_dct = pickle.load(fp)                        

        if self.ion_family =='moderate':
            x_,PDF_ = np.loadtxt(__path__+'/''docs/pdf_moderate.txt',unpack=True)
        elif  self.ion_family =='low':
            x_,PDF_ = np.loadtxt(__path__+'/''docs/pdf_low.txt',unpack=True)
        elif  self.ion_family =='user':
            x_,PDF_ = np.loadtxt(self.filename_ion_family,unpack=True)
        
        #store distribution of delta v 
        self.x=x_
        self.PDF=PDF_

        
        #here read oscillator strengths from linetools or file
        if self.filename_ion_list == None:
            from linetools.lists.linelist import LineList
            self.lines = LineList('Strong')
                        
            self.fstren = np.zeros_like(self.index, dtype=np.float)
            for ind in np.arange(self.nsim):
                self.fstren[ind] = self.lines[self.trans_wl[ind]*u.AA]['f']
                
            self.fstren=np.array(self.fstren)
            if(self.fstren.any() is None):
                raise ValueError('Wavelengths not found in database. Aborting')
                exit()
        else:
            self.line_info= pd.read_csv(self.filename_ion_list,comment = '#',delim_whitespace=True)
            raise Exception('This needs more coding to import the oscillator strengths in an array. Sorry')
            exit()


        # Set seed if needed
        if self.seed != None:
            np.random.seed(self.seed)
            random.seed(self.seed)
                   
        #Define native pixel scale (1km/s)
        #The shape of C in the NMF which is 1198
        #Pix 599 is at zero 
        self.native_pix=1.0
    
   
        #get shape of vel array in native pixels
        self.natvpix=self.NMF_dct[0]['C'].shape[1]
        #index of zero velocity in native pixels
        self.natvzero=np.int(self.natvpix/2)



    def _check_params(self):
        
        """
        This function checks for input paramaters and suggests how to fix errors
 
        """


        #check n_sim is postivive integer         
        if (
            not isinstance(self.nsim, numbers.Integral)
            or self.nsim <= 0
        ):
            raise ValueError(
                "Number of simulations must be a positive integer; got "
                f"(nsim={self._nsim!r})"
            )
    

    
        # ion_family
        allowed_family = ("moderate", "low", "user")
        if self.ion_family not in allowed_family:
            raise ValueError(
                f"Invalid ion_family parameter: got {self.ion_family!r} instead of one of "
                f"{allowed_family}"
            )

            
        #keep sims and parameters of equal number
        if self.nsim != len(self.trans_wl):
            raise ValueError("nsim and trans_wl do not match in size")
        
        if self.nsim != len(self.ion):
            raise ValueError("nsim and ion do not match in size")
            
        if self.nsim != len(self.ion_logN):
            raise ValueError("nsim and ion_logN do not match in size")
            
        if (self.SN.any() != None) & (self.nsim != len(self.SN)):
            raise ValueError("nsim and SN do not match in size")
                
        return 
        
    def _check_files(self):

        """
        This function checks input files 

        """


        # filename_ion_family
        if (
            self.ion_family == 'user'
            and self.filename_ion_family == None
        ):
            raise ValueError(
                "Got "f" ion_family={self.ion_family!r} " "but you did not provide a file for DeltaV_90 pdf "
            )
            
        return 
        
        
    
    def _count_intervals(self, sequence, intervals):
        
        """

        Utility function to count intervals in delta velocity distribution

        """

        
        count = defaultdict(int)
        
        intervals.sort()
        for item in sequence:
            pos = bisect_left(intervals, item)
            if pos == len(intervals):
                count[None] += 1
            else:
                count[intervals[pos]] += 1
        
        if all(elem is None for elem in [k for k in count.keys()]):
            count = 0
            
        else:
         
            if (max({k for k in count.keys() if k is not None})) == intervals[1] :
                count = count[max({k for k in count.keys() if k is not None})]
            else:
                count = 0
        
    
        return count
            
            
        
    def _NMF_profile(self, X, C, nsim_bin):
        """Generate a profile based on the non-negative matrices X, C stored in NMF_dtc
        
        Parameters
        ----------
        X : NMF_dct['X'] ndarray of shape n × m, where m is the number of reduced features in the NMF space
        
        C : NMF_dct['C'] ndarray of shape m × u representing the basis matrix of the m reduced features
        
        Returns
        -------
        
        simulated : ndarray of shape (n_sim, u) simulated profiles
        
        """
            
        simulated =[]
        
        # Number of realization to be performed: set to 1 + the ratio between the total number of simulations and the number of data in the bin
        n_data, n_features = X.shape
        
        nreal = int(nsim_bin/n_data) + 1
    
        for n in range(nreal):
        
    
            simulated_NMF_space_tmp=np.zeros((int(n_data),(n_features)))
            for ic in range(n_features):
                simulated_NMF_space_tmp[:,ic] = random.sample(list(X[:,ic]),int(n_data))
        
            simulated_tmp = np.dot(simulated_NMF_space_tmp,C)
            simulated.append(simulated_tmp)
       
        simulated = [x for xs in simulated for x in xs]
    
        return simulated[0:nsim_bin]

        
    def _compute_ew(self,wave,flux):


        """
        Utility function to compute the EW of the profile

        """
    
        delta_w=np.roll(wave,-1)-wave
        ew=np.sum((1-flux[:,:-1])*delta_w[:,:-1],axis=1)
        
        return ew
        
        
    def simulation(self):
           
        """
        This is the main function to calls profile making and spectra making routins
        
        """
        
        if(self.verbosity>0):
            print("NMF-PM: starting profile simulations")            
            
        #generate profiles
        S=self._run_profiles()


        if(self.verbosity>0):
            print("NMF-PM: now compute observed profiles in flux space")
        
        #turn profiles into spectra 
        MN=self.metal(S)
        

        if(self.verbosity>0):
            print("NMF-PM: All done!")
            
        return
    

    def _run_profiles(self):
        
    
        """
        This function controls the workflow to draw profiles
        
        """


        #get time 
        start_time = time.time()
        
        #draw from DV90 distribution
        DV90_dist = np.random.choice(self.x,self.nsim, p=self.PDF)
    
        #now generate the profiles in velocity space
        edges = [self.NMF_dct[i]['edg'] for i in range(len(self.NMF_dct))]
        
        S = []
        count = 0
        for j in range(len(edges)):
            nsim_bin = self._count_intervals(DV90_dist,edges[j])

            if nsim_bin > 0:
                if (self.NMF_dct[j]['V_comp'] == 'all'):
                    count+= nsim_bin
                if (self.NMF_dct[j]['V_comp'] == 'low') or (self.NMF_dct[j]['V_comp'] == 'high'):
                    nsim_bin = int(nsim_bin/2) + 1
                        
                X = self.NMF_dct[j]['X']
                C = self.NMF_dct[j]['C']
                
                profiles_sim_bin = self._NMF_profile(X,C,nsim_bin)
                S.append(profiles_sim_bin)
        
        S = [x* (10**(-12)) for xs in S for x in xs]
        
        index_del = [random.randint((self.nsim-count), len(S)-1) for i in range(len(S)-self.nsim)]
        if len(index_del) >=1:
            S = np.delete(S,index_del,axis=0)
            
        if self.verbosity > 0:
            print("NMF-PM: it takes", time.time() - start_time, "to simulate", self.nsim, "velocity profiles")
    
        return np.array(S)
            

    def metal(self,S):

        """

        This function turns aborption profiles into spectra
      
        """

        #speed light in km/s
        _c = 299792458e-3 
        
        if(self.verbosity > 0):
            print('NMF-PM: Creating spectrum in flux space')
            
        #compute the normalization part in vector form
        cne = 0.014971475*np.sqrt(np.pi)*(10.**self.ion_logN)*self.fstren
        metals1 = np.exp(-1.*S*cne[:,np.newaxis])
        
        #compute second term of the doublet if so desired
        if self.doublet:
            metals2 = np.exp(-1.*S*cne[:,np.newaxis]*self.dbl_fratio)
            
            #compute new length of array to fit in doublet 
            new_index=np.int(2*self.natvzero+self.dbl_dvel)
            vel_native=np.arange(new_index)-self.natvzero
            
            #insert first profile
            metals=np.ones((self.nsim,new_index))
            metals[:,0:self.natvpix]=metals[:,0:self.natvpix]*metals1


            if(self.verbosity > 0):
                print('NMF-PM: Inserting doublets')

            #insert second profile
            istart=np.int(self.dbl_dvel)
            iend=istart+self.natvpix
            metals[:,istart:iend]=metals[:,istart:iend]*metals2
            

        else:
            #if no doublet, use original size for wave determination 
            vel_native=np.arange(self.natvpix)-self.natvzero
            metals=metals1

        #store velocity,wave and flux 
        self.velocity=vel_native
        self.flux=np.array(metals)        
        #self.wave=np.outer(self.trans_wl,self.velocity/_c)+self.trans_wl.reshape(-1,1)
        
        
        if self.convolved == True:
            if(self.verbosity > 0):
                print('NMF-PM: Applying convolution kernel')
                start_time=time.time()

            #define kernel and convolve 
            self.gauss_kernel = Gaussian1DKernel(stddev=self.res/2.355)
            self.flux = convolve1d(self.flux, self.gauss_kernel.array, axis = 1)
            self.flux=np.array(self.flux)
           
            if self.verbosity > 0:
               print("NMF-PM: it takes", time.time() - start_time, "to convolve", self.nsim, "velocity profiles")
        
        if self.px_scale != None:
            
            if self.verbosity > 0:
                print("NMF-PM: Re-binning to desired pixel scale")
                
            if np.modf(self.px_scale/self.native_pix)[0] == 0:
                if self.verbosity >0:
                    print("NMF-PM: Running non interpolating rebinning")

                #compute the shape of rebinning 
                s0, s1 = np.shape(self.flux)
                self.rebinfac   = int(self.px_scale/self.native_pix)
                self.finalNpix  = int(s1//self.rebinfac)

                
                #rebin flux
                self.flux = np.mean((self.flux[:,:self.finalNpix*self.rebinfac]).reshape(s0, self.finalNpix, self.rebinfac), axis=2)

                #now work on velocity 
                self.vel_native = self.velocity[:self.finalNpix*self.rebinfac]
                self.velocity  = np.interp(np.arange(self.finalNpix)*self.rebinfac+self.rebinfac/2., np.arange(len(self.vel_native)),
                                           self.vel_native-self.native_pix/2)

                               
                
            else:
                if self.verbosity >0:
                    print("NMF-PM: Running fractional pixel rebinning")
                
                #First find greatest common divisor between target pix scale and native one with precision of 0.1kms
                
                rnd_px_scale = np.round(self.px_scale, decimals=1)
                if rnd_px_scale != self.px_scale:
                   if self.verbosity>0:
                     print("NMF-PM: Target pixel scale of {:3.5f} kms has been round to {:3.1f} kms".format(self.px_scale, rnd_px_scale))
                   self.px_scale = rnd_px_scale
                
                precision = 0.1
                
                px_common = np.gcd(int(self.px_scale/precision),int(self.native_pix/precision))*precision
                oversample = int(self.native_pix/px_common)
                
                oversamp_grid = np.linspace(np.min(self.velocity), np.max(self.velocity), num=(len(self.velocity)*oversample))
                                
                flux_interp = interp1d(self.velocity, self.flux, axis=1, kind='nearest-up')(oversamp_grid)
                
                s0, s1 = np.shape(flux_interp)
                self.rebinfac   = int(self.px_scale/px_common)
                self.finalNpix  = int(s1//self.rebinfac)

                #rebin flux
                self.flux      = np.mean((flux_interp[:,:self.finalNpix*self.rebinfac]).reshape(s0, self.finalNpix, self.rebinfac), axis=2)
                self.velocity  = np.interp(np.arange(self.finalNpix)*self.rebinfac+self.rebinfac/2., np.arange(len(oversamp_grid)),
                                           oversamp_grid-px_common/2.)

                        
        self.wave=np.outer(self.trans_wl,self.velocity/_c)+self.trans_wl.reshape(-1,1)

        #store no noise spectra
        self.flux_nonoise = np.copy(self.flux)

        #omage de la maison, ew
        self.ew=self._compute_ew(self.wave,self.flux)
        
        if self.SN.any()!= None:

            if self.verbosity > 0:
                print("NMF-PM: Inject noise into spectra")

            #call noise generator
            self._gnoise_add()

            
    def _gnoise_add(self):

        """
        Function that handles the noise generation

        """

        #draw a poisson from flux 
        _poisson = np.random.poisson(self.flux * ((self.SN**2.)[:,np.newaxis]))
        noise_continuum = np.sqrt(_poisson)

        #if so desired add sky noise
        if self.sigma_sky != None:
            noise_sky = np.random.normal(0, self.sigma_sky, size=np.shape(noise_continuum))
            noise_array= np.sqrt(noise_continuum**2. +noise_sky**2.)
        else:
            noise_array= noise_continuum
            noise_sky=0

        #compute and store 
        profile_noise = _poisson + noise_sky
        self.flux = profile_noise/((self.SN**2.)[:,np.newaxis])
        self.noise = noise_array/((self.SN**2.)[:,np.newaxis])
        
        return        
        
    

def main():
    print(__doc__)

if __name__=='__main__':
    main()
        
