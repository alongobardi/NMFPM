import os
import numbers
import numpy as np
import pickle
import time
import pandas as pd
import random
from astropy.convolution import convolve, Gaussian1DKernel
from collections import defaultdict
from bisect import bisect_left


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
        
    seed = int, default = None
    
    verbosity = int, defalut = 0
        Print (1) or not (0) info to terminal
    
    
    Attributes
    ----------

    metals_ = Library of nsim synthetic metals
    metals_nonoise_ = Library of nsim synthetic metals, noise free 
    noise_ = Array of noise values for the nsim synthetic metals
    wavelength_ = Array of wavelength values for the nsim synthetic metals
    
    
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
        SN = np.array([None]),
        sigma_sky = None,
        seed = None,
        verbosity= 0
        
    
    
    ):
        
        self.nsim = nsim
        self.ion_family = ion_family
        self.filename_ion_family = filename_ion_family
        self.ion_logN = ion_logN
        self.ion = ion
        self.trans_wl = trans_wl
        self.filename_ion_list = filename_ion_list
        self.convolved = convolved
        self.res = res
        self.px_scale = px_scale
        self.SN = SN
        self.sigma_sky = sigma_sky
        self.seed = seed
        self.verbosity=verbosity
        
        
        
        # Run saftey checks
        # Check parameters
        self._check_params()
        # Check files
        self._check_files()
        
        
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
        
        self.x=x_
        self.PDF=PDF_


        if self.filename_ion_list == None :
            line_info= pd.read_csv(__path__+'/''docs/linelist.ascii.txt',comment = '#',delim_whitespace=True)
            self.line_info = line_info[(line_info.gamma != 0) & (~line_info['gamma'].isnull())]
        else:
            self.line_info= pd.read_csv(self.filename_ion_list,comment = '#',delim_whitespace=True)
            
        # Set seed if needed
        if self.seed != None:
            np.random.seed(self.seed)
            random.seed(self.seed)
                   
        # Define native pixel scale (1km/s)
        self.native_pix=1.002
        
   
    def _check_params(self):
        # n_sim
        self._nsim = self.nsim
        
        if (
            not isinstance(self._nsim, numbers.Integral)
            or self._nsim <= 0
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
                
        
        return self
        
    def _check_files(self):
        # filename_ion_family
        if (
            self.ion_family == 'user'
            and self.filename_ion_family == None
        ):
            raise ValueError(
                "Got "f" ion_family={self.ion_family!r} " "but you did not provide a file for DeltaV_90 pdf "
            )
            
        return self
        
        
    
    def _count_intervals(self, sequence, intervals):
        
    
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
    
    def _convolve(self,S):
        
        
        gauss_kernel = Gaussian1DKernel(stddev=self.res/2.355)
        conv = convolve(S,gauss_kernel,boundary='extend')
        
        return conv
        
    
    def _resample_matrix(self,orig_spec_axis,fin_spec_axis):
        #Create a re-sampling matrix to be used in re-sampling spectra in a way that conserves flux
        orig_edges = orig_spec_axis
        fin_edges = fin_spec_axis[(fin_spec_axis > orig_edges[0]) & (orig_edges[-1] > fin_spec_axis)]
        step = 1
        while step <=2:
            
            # Lower bin and upper bin edges
            orig_low = orig_edges[:-1]
            fin_low = fin_edges[:-1]
            orig_upp = orig_edges[1:]
            fin_upp = fin_edges[1:]
        
            
            l_inf = np.where(orig_low > fin_low[:, np.newaxis],
                            orig_low, fin_low[:, np.newaxis])

            l_sup = np.where(orig_upp < fin_upp[:, np.newaxis],
                            orig_upp, fin_upp[:, np.newaxis])
        
        
            
            resamp_mat = (l_sup - l_inf).clip(0)
            
            resamp_mat = resamp_mat * (orig_upp - orig_low)

            left_clip = np.where(fin_edges[:-1] - orig_edges[0] < 0, 0, 1)
            
            right_clip = np.where(orig_edges[-1] - fin_edges[1:] < 0, 0, 1)
                
            keep_overlapping_matrix = left_clip * right_clip
        
            resamp_mat *= keep_overlapping_matrix[:, np.newaxis]
            bin_size = np.sum(resamp_mat, axis=-1)

            
            step +=1
                
            # Lower bin and upper bin edges
            const_ =  (np.abs((bin_size) - self.native_pix)/2.)
           
            fin_low = fin_edges[:-1]- const_
            fin_upp = fin_edges[1:]- const_
        
            l_inf = np.where(orig_low > fin_low[:, np.newaxis],
                            orig_low, fin_low[:, np.newaxis])

            l_sup = np.where(orig_upp < fin_upp[:, np.newaxis],
                            orig_upp, fin_upp[:, np.newaxis])
        
        
                
            resamp_mat = (l_sup - l_inf).clip(0)
            resamp_mat = resamp_mat * (orig_upp - orig_low)

            left_clip = np.where((fin_edges[:-1]- const_) - orig_edges[0] < 0, 0, 1)
            right_clip = np.where(orig_edges[-1] - (fin_edges[1:]- const_) < 0, 0, 1)
            keep_overlapping_matrix = left_clip * right_clip
        
            resamp_mat *= keep_overlapping_matrix[:, np.newaxis]

            return resamp_mat
            
    def _resampling(self, S, resample_grid):
        
        origin_spec_axis = np.arange(-600,600,self.native_pix)
        final_spec_axis = np.arange(-600,600,self.px_scale)
        
        
        new_flux_shape = list(S.shape)
        new_flux_shape.insert(-1, 1)

        in_flux = S.reshape(new_flux_shape)
        ones = [1] * len(S.shape[:-1])
        new_shape_resample_grid = ones + list(resample_grid.shape)
        resample_grid = resample_grid.reshape(new_shape_resample_grid)


        out_flux =  [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*resample_grid.T)] for X_row in in_flux]/ np.sum(resample_grid, axis=-1)
        resampled_spectrum = out_flux[0]
        
        y2 = np.interp(final_spec_axis,final_spec_axis[(final_spec_axis>  origin_spec_axis[0]) & ( origin_spec_axis[-1] > final_spec_axis)][:-1],resampled_spectrum)
        
        
        return y2
        
        
    def _NMF_profile(self, X, C, nsim_bin):
        """Generate a profile based on the non-negative matrices X, C stored in NMF_dtc
        
        Parameters
        ----------
        X : NMF_dct['X'] ndarray of shape 𝑛 × 𝑚, where 𝑚 is the number of reduced features in the NMF space
        
        C : NMF_dct['C'] ndarray of shape 𝑚 × 𝑣 representing the basis matrix of the 𝑚 reduced features
        
        Returns
        -------
        
        simulated : ndarray of shape (n_sim, 𝑣) simulated profiles
        
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
        
        
    def simulation(self):
        
        
        start_time = time.time()
        
        DV90_dist = np.random.choice(self.x,self.nsim, p= self.PDF)
    
    
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
        
        
        
        MN=self.metal(S)
        self.metals_ = MN[0]
        self.noise_ = MN[1]
        self.metals_nonoise_ = MN[2]
        self.wavelength_ = self.wavel()
        
        if self.verbosity > 0:
            print("NMF-PM takes", time.time() - start_time, "to simulate", self.nsim, "profiles")
        return S
        
    def metal(self,S):
        
        trans_os=np.zeros(len(self.ion))
        
        
        for j in range(len(self.ion)):
            
            df_tmp = self.line_info[[x.split( )[0] == self.ion[j] for x  in self.line_info.name.values]].reset_index(drop=True)
            
            # Consider transitions with wrest within 0.5 A from self.trans_wl.
            # If mutiple transitions satisfy this condition, select the transition with the smallest difference between wrest and self.trans_wl
            df_tmp = df_tmp.loc[[np.abs(df_tmp.wrest.values[i] - self.trans_wl[j]) < 0.5 for i in range(len(df_tmp))]]
            
            if len(df_tmp) == 1:
                trans_os[j] = df_tmp.f.values
            
            else:
                wl_dif = np.abs([df_tmp.wrest.values[i] - self.trans_wl[j] for i in range(len(df_tmp))])
                trans_os[j] = df_tmp.loc[wl_dif == np.min(wl_dif), "f"].values
            
            
        cne = [0.014971475 * np.sqrt(np.pi)* (10.**self.ion_logN[i]) * trans_os[i] for i in range(self.nsim)]
        
        metals = np.exp(-1.* np.array([cne[i]* S[i] for i in range(self.nsim)]))
        
        
        if self.convolved == True:
            metals = [self._convolve(m) for m in metals]
        
        
        if self.px_scale != None:
            origin_spec_axis = np.arange(-600,600,self.native_pix)
            final_spec_axis = np.arange(-600,600,self.px_scale)
            
            resample_grid =self._resample_matrix(origin_spec_axis,final_spec_axis)
            
            metals = [self._resampling(mm,resample_grid) for mm in metals ]

        
       
        if self.SN.any() != None:
            noise=[]
            metalnoise=[]
            ii=0
            for mm in metals:
                metwnoise=self.gnoise_add(mm,self.SN[ii])
                noise.append(metwnoise[1])
                metalnoise.append(metwnoise[0])
                ii=ii+1
            return metalnoise,noise,metals
        else:
            return metals,None,metals
        
    def wavel(self):
    
        _c = 299792458e-3 # km/s
        wavelength = []
        for j in range(len(self.ion)):
        
            if self.px_scale != None:
                vel=np.arange(-600,600,self.px_scale)
            else:
                vel=np.arange(-600,600,self.native_pix)
                
            wavelength.append((np.array(vel)/_c * self.trans_wl[j])+self.trans_wl[j])
    
        return wavelength
        
    def gnoise_add(self,m,SN):
        
        
        _poisson = np.random.poisson(m * (SN**2.))
        noise_continuum = np.sqrt(_poisson)
        
        if self.sigma_sky != None:
            noise_sky = np.random.normal(0, self.sigma_sky, len(m))
            noise_array= np.sqrt(noise_continuum**2. +noise_sky**2.)
        else:
            noise_array= noise_continuum
            noise_sky=0
            
        profile_noise = _poisson + noise_sky
        
        return profile_noise/(SN**2.),noise_array/(SN**2.)
        
        
    

def main():
    print(__doc__)

if __name__=='__main__':
    main()
        
