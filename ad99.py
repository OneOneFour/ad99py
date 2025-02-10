from scipy.integrate import cumulative_trapezoid,trapezoid
import numpy as np


## CONSTANTS
g = 9.81
Rd = 287.04
c_p = 7 * Rd / 2

## Class implementation


class AlexanderDunkerton1999:
    def __init__(
        self,
        source_level_height=9e3,
        Bm=0.4,
        cw=35,
        Fs0=0.004,
        damp_top=False,
        cmax=99.9,
        dc=0.5,
        base_wavelength=300e3,
        force_intermittency=None,
    ):
        """
        Initialize an AlexanderDunkerton1999 Non-orographic drag parameterization instance.
        Following Alexander & Dunkerton 1999 (Journal of the Atmospheric Sciences)

        Parameterization Parameters:
        Bt: Total Equatorial GW Momentum Flux (m^2/s^2)
        cw: Half Width at half maximum of GW source spectrum (m/s)

        Grid Parameters:
        pfull - Full pressure levels (hPa) [N]
        rho - Density on full pressure levels (kg/m^3) [N]


        TODO:
        """
        self.cmax = cmax
        self.dc = dc
        self.Bm = Bm
        self.cw = cw
        self.cmax = cmax
        self.dc = dc
        self.Fs0 = Fs0
        self.use_intrinsic_c = True
        self.source_level_height = source_level_height
        self.damp_top = damp_top
        ## Could be increased to consider higher wave vectors but for simplicity nk=1
        self.base_wavelength = base_wavelength
        self.force_intermittency = force_intermittency  

        self.c0 = np.arange(-self.cmax, self.cmax + self.dc, self.dc)
        self.kwv = (2 * np.pi/self.base_wavelength)
 

    def source_spectrum(self, c, u=None):
        if self.use_intrinsic_c:
            c = c - u
        return np.sign(c) * self.Bm *np.exp(-np.log(2) * (c / self.cw) ** 2)

    def intermittency(self, rho_source, u=None):
        if self.force_intermittency:
            return self.force_intermittency
        return (self.Fs0 * self.dc) / (
            rho_source * trapezoid(np.abs(self.source_spectrum(self.c0, u)), dx=self.dc)
        )

    def reflection_frequency(self,N,H):
        alpha = 1/(2*H)
        return np.sqrt( N**2 * self.kwv**2/ (self.kwv**2 + alpha*alpha))

    def gwd(self, u, N, z,rho):
        """
        Input:
        u - zonal wind profile (m/s) at each height level [...,N]
        N - Buoyancy frequency (s^-1) at each height level [...,N]

        Output:
        GWD - Non orographic Gravity wave drag at each height level [...,N]

        """
        drag = np.zeros_like(u)
        ## estimate idx of source level 
        idx_z_source = np.argmin(np.abs(z - self.source_level_height))
        # Slice arrays to start from the array 
        z_sources = z[idx_z_source:]
        u_sources = u[idx_z_source:]
        N_sources = N[idx_z_source:]    
        rho_sources = rho[idx_z_source:]

        # Wave mask of unbroken waves
        wave_mask = np.ones_like(self.c0,dtype=bool)

        ## Source level parameters 
        rho_0 = rho_sources[0]
        u_0 = u_sources[0]
        source_spectrum = self.source_spectrum(self.c0,u_0)

        for i,(z_lvl,u_lvl,N_lvl,rho_lvl) in enumerate(zip(z_sources,u_sources,N_sources,rho_sources)):
            intrinsic_freq = self.kwv*(self.c0 - u_lvl)
            # Total internal reflection
            H = -(z_lvl - z[idx_z_source + i-1])/np.log(rho_lvl/rho[idx_z_source + i-1])
            tir = np.abs(intrinsic_freq) < self.reflection_frequency(N_lvl,H)
            wave_mask = wave_mask & tir
            # Breaking condition
            if not wave_mask.any():
                break
            Q0 =2*N_lvl*source_spectrum*rho_0/(rho_lvl*self.kwv*(self.c0 - u_lvl)**3)

            if i > 0: 
                # Above breaking level check for losses 
                if i == len(z_sources) - 1 and self.damp_top: 
                    breaking_waves = wave_mask
                else:
                    breaking_waves = wave_mask & (Q0 >= 1) 
                dz = z_lvl - z_sources[i-1]
                F0 = np.sum(rho_0*source_spectrum[breaking_waves])
                rho_half = np.sqrt(rho_lvl*rho[idx_z_source + i-1])
                X_half = F0*self.intermittency(rho_0,u_0)/(rho_half*dz)
                drag[idx_z_source + i] = X_half 
                drag[idx_z_source + i - 1] = 0.5*(X_half + drag[idx_z_source + i-1])


                # # Last level    
                # breaking_waves = wave_mask 
                # dz = z_lvl - z_sources[i-1] 
                # F0 = np.sum(rho_0*self.source_spectrum(self.c0[breaking_waves],u_0))    
                # rho_half = np.sqrt(rho_lvl*rho[idx_z_source + i-1])
                # X_half = F0*self.intermittency(rho_0,u_0)/(rho_half*dz)
                # drag[-1] = X_half
                # drag[-2] = drag[-2] * 0.5 

            wave_mask = wave_mask & (Q0 < 1 )
        return drag 

    def gwd_momentum_flux(self, gwd,z,rho):
        uw = -1*cumulative_trapezoid(rho[::-1]*gwd[::-1],x=z[::-1],initial=0)
        return uw



    
    def gwd_momentum_flux_ptv(self,gwd,z,rho):
        """
        Calculate the momentum flux due to gravity waves only 
        """
        gwd_ptv = np.where(gwd > 0,gwd,0)
        


        # Integrate from top to bottom 
        # We are working on height levels here (integral is dz not dp/g )
        uw = -1*cumulative_trapezoid(rho[::-1]*gwd_ptv[::-1],x=z[::-1],initial=0)[::-1]
        return uw


    def gwd_momentum_flux_ntv(self,gwd,z,rho):
        """
        Calculate the momentum flux due to gravity waves only 
        """
        drag_ntv = np.where(gwd < 0, gwd, 0)

        # Integrate from top to bottom 
        # We are working on height levels here (integral is dz not dp/g )
        uw = -1*cumulative_trapezoid(rho[::-1]*drag_ntv[::-1],x=z[::-1],initial=0)[::-1] 
        return uw