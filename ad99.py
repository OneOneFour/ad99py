from scipy.integrate import cumulative_trapezoid, trapezoid
import numpy as np


## CONSTANTS


## Class implementation


class AlexanderDunkerton1999:
    def __init__(
        self,
        source_level_height=9e3,
        Bm=0.4,
        cw=35,
        Fs0=0.004,
        damp_top=False,
        cmax=99.6,
        dc=1.2,
        base_wavelength=300e3,
        force_intermittency=None,
        use_intrinsic_c=True,
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
        self.use_intrinsic_c = use_intrinsic_c
        self.source_level_height = source_level_height
        self.damp_top = damp_top
        ## Could be increased to consider higher wave vectors but for simplicity nk=1
        self.base_wavelength = base_wavelength
        self.force_intermittency = force_intermittency

        self.c0 = np.arange(-self.cmax, self.cmax + self.dc, self.dc)
        self.kwv = 2 * np.pi / self.base_wavelength

    def __repr__(self):
        return (f"AlexanderDunkerton1999(Bm={self.Bm},cw={self.cw}"
                f",Fs0={self.Fs0},height={self.source_level_height},"
                f" λ={self.base_wavelength}, damp_top={self.damp_top})")

    def source_spectrum(self, c, u=None):
        c0 = u if self.use_intrinsic_c else 0
        return np.sign(c-u) * self.Bm * np.exp(-np.log(2) * ((c-c0)/ self.cw) ** 2)

    def intermittency(self, rho_source, u=None):
        if self.force_intermittency:
            return self.force_intermittency
        return (self.Fs0 * self.dc) / (
            rho_source * trapezoid(np.abs(self.source_spectrum(self.c0, u)), dx=self.dc)
        )

    def intermittency_dask(self, rho_source, u=None):
        import dask.array as da

        if self.force_intermittency:
            return self.force_intermittency
        return (self.Fs0 * self.dc) / (
            # Technically this is a stricter implementation of the AD99 definition of eps
            rho_source
            * da.sum(da.abs(self.source_spectrum(self.c0, u)) * self.dc, axis=-1)
        )

    def reflection_frequency(self, N, H):
        alpha = 1 / (2 * H)
        return np.sqrt(N**2 * self.kwv**2 / (self.kwv**2 + alpha * alpha))

    # def vectorized_gwd_2(self,u,N,z,rho):
    #     drag = np.zeros_like(u)
    #     idx_z_source = np.argmin(np.abs(z - self.source_level_height),axis=-1,keepdims=True)

    #     rho_0 = np.take_along_axis(rho,idx_z_source,axis=-1)
    #     u_0 =  np.take_along_axis(u,idx_z_source,axis=-1)
    #     source_spectrum = self.source_spectrum(self.c0,u_0)
    #     dz = z[...,1:] - z[...,:-1]
    #     H = -(dz/np.log(rho[...,1:]/rho[...,:-1]))
    #     reflection_frequency = self.reflection_frequency(N[1:],H)
    #     intrinsic_freq = self.kwv*(self.c0[...,:,None] - u[...,None,:])
    #     wave_mask = np.ones_like(source_spectrum,dtype=bool)
    #     intermittency = self.intermittency(rho_0,u_0)
    #     for i in range(np.min(idx_z_source),z.shape[-1]):

    #         tir = np.abs(intrinsic_freq[...,i]) < reflection_frequency[...,i]
    #         wave_mask = wave_mask & tir
    #         Q0 = 2*N[...,i]*source_spectrum*rho_0/(rho[...,i]*self.kwv*(self.c0 - u[...,i])**3) * (i >= idx_z_source)
    #         breaking_wave = wave_mask & (Q0 * (i > idx_z_source) >= 1)
    #         F0 = np.sum(rho_0*source_spectrum[breaking_wave],axis=-1) ## batch_shape
    #         rho_half = np.sqrt(rho[...,i]*rho[...,i-1])

    #         drag[...,i] = F0*intermittency/(rho_half*dz)
    #         drag[...,i-1] = 0.5*(drag[...,i] + drag[...,i-1])

    #         wave_mask = wave_mask & (Q0 < 1)

    #     return drag

    # def vectorized_gwd(self, u, N, z,rho):
    #     drag = np.zeros_like(u)
    #     idx_z_source = np.argmin(np.abs(z - self.source_level_height),axis=-1,keepdims=True)

    #     rho_0 = np.take_along_axis(rho,idx_z_source,axis=-1)
    #     u_0 =  np.take_along_axis(u,idx_z_source,axis=-1)
    #     source_spectrum = self.source_spectrum(self.c0,u_0)

    #     intrinsic_freq = self.kwv*(self.c0[...,:,None] - u[...,None,:])
    #     dz = z[...,1:] - z[...,:-1]
    #     H = -(dz/np.log(rho[...,1:]/rho[...,:-1]))
    #     tir = np.abs(intrinsic_freq) < self.reflection_frequency(N,H)[...,None,:]
    #     tir_idx = np.where(tir.any(axis=-1,keepdims=True) , np.argmax(tir,axis=-1,keepdims=True), -1)

    #     Q0 = (2*N*source_spectrum[...,:,None]*rho_0/(rho[...,None,:]*self.kwv*(self.c0 - u[...,None,:])**3) >= 1) # batch_shape, c,z
    #     breaking_idx = np.where(Q0.any(axis=-1,keepdims=True), np.argmax(Q0,axis=-1,keepdims=True), Q0.shape[-1] -1 )
    #     breaking_idx = np.where(breaking_idx < tir_idx, breaking_idx, -1)

    #     s

    #     drag_at_level = np.sum(rho_0* source_spectrum*deposit, axis=-2)

    #     np.sum( rho_0)

    #     return drag

    def gwd_vectorized(self, u, N, z, rho):
        """
        Vectorized implementation
        """
        level_idx = np.broadcast_to(np.arange(z.shape[-1], dtype=int), z.shape)
        idx_z_source = np.argmin(
            np.abs(z - self.source_level_height), axis=-1, keepdims=True
        )
        rho_0 = np.take_along_axis(rho, idx_z_source, axis=-1).squeeze(axis=-1)
        u_0 = np.take_along_axis(u, idx_z_source, axis=-1).squeeze(axis=-1)
        spectrum = self.source_spectrum(self.c0, u_0[..., None])

        dz = z[..., 1:] - z[..., :-1]
        H = -(dz / np.log(rho[..., 1:] / rho[..., :-1]))

        dz = np.concatenate([dz[..., 0, None], dz], axis=-1)
        H = np.concatenate([H[..., 0, None], H], axis=-1)

        reflection_frequency = self.reflection_frequency(N, H)
        intrinsic_freq = self.kwv * (self.c0[..., :, None] - u[..., None, :])
        eps = self.intermittency(rho_0, u_0)
        tir = (np.abs(intrinsic_freq) >= reflection_frequency[..., None, :]) * (
            (level_idx >= idx_z_source)[..., None, :]
        )
        tir_idx = np.where(
            tir.any(axis=-1, keepdims=True),
            np.argmax(tir, axis=-1, keepdims=True),
            tir.shape[-1],
        )

        Q0 = (2 * N[..., None, :] * spectrum[..., :, None] * rho_0[..., None, None]) / (
            rho[..., None, :]
            * self.kwv
            * (self.c0[..., :, None] - u[..., None, :]) ** 3
        )  # batch_shape, c,z
        valid_starting_waves = (
            (idx_z_source == level_idx)[..., None, :] * (Q0 < 1)
        ).any(axis=-1, keepdims=True)
        breaking_waves = (
            (Q0 >= 1) * (level_idx > idx_z_source)[..., None, :] * valid_starting_waves
        )

        breaking_idx = np.where(
            breaking_waves.any(axis=-1, keepdims=True),
            np.argmax(breaking_waves, axis=-1, keepdims=True),
            Q0.shape[-1] * self.damp_top - 1,
        )
        breaking_idx = np.where(breaking_idx < tir_idx, breaking_idx, -1)
        breaking = (breaking_idx == level_idx)[..., None, :]
        f0 = np.sum(breaking * rho_0[..., None, None] * spectrum[..., None], axis=-2)
        rho_half = np.concatenate(
            [rho[..., 0], np.sqrt(rho[..., 1:] * rho[..., :-1])], axis=-1
        )
        half_lvl_drag = f0 * eps[..., None] / (rho_half * dz)
        full_lvl_drag = np.concatenate(
            [
                0.5 * (half_lvl_drag[..., 1:] + half_lvl_drag[..., :-1]),
                half_lvl_drag[..., -1, None],
            ],
            axis=-1,
        )
        return full_lvl_drag

    def gwd_dask(self, u, N, z, rho):

        def take_along_axis(data, idx):
            return np.take_along_axis(data[0], idx[0], axis=-1)

        import dask.array as da

        """
        vectorized dask friendly implementation
        """

        level_idx = da.broadcast_to(
            da.arange(z.shape[-1], dtype=int), z.shape, chunks=z.chunks
        )
        idx_z_source = da.argmin(
            da.abs(z - self.source_level_height), axis=-1, keepdims=True
        )
        rho_0 = da.blockwise(
            take_along_axis, "ij", rho, "ijk", idx_z_source, "ijk", dtype=rho.dtype
        )
        u_0 = da.blockwise(
            take_along_axis, "ij", u, "ijk", idx_z_source, "ijk", dtype=u.dtype
        )
        spectrum = self.source_spectrum(self.c0, u_0[..., None])

        dz = z[..., 1:] - z[..., :-1]
        H = -(dz / da.log(rho[..., 1:] / rho[..., :-1]))

        dz = da.concatenate([dz[..., 0, None], dz], axis=-1)
        H = da.concatenate([H[..., 0, None], H], axis=-1)
        dz = dz.rechunk(z.chunks)
        H = H.rechunk(z.chunks)

        reflection_frequency = self.reflection_frequency(N, H)
        intrinsic_freq = self.kwv * (self.c0[..., :, None] - u[..., None, :])
        eps = self.intermittency_dask(rho_0, u_0)
        tir = (da.abs(intrinsic_freq) >= reflection_frequency[..., None, :]) * (
            (level_idx >= idx_z_source)[..., None, :]
        )
        tir_idx = da.where(
            tir.any(axis=-1, keepdims=True),
            da.argmax(tir, axis=-1, keepdims=True),
            tir.shape[-1],
        )

        Q0 = (2 * N[..., None, :] * spectrum[..., :, None] * rho_0[..., None, None]) / (
            rho[..., None, :]
            * self.kwv
            * (self.c0[..., :, None] - u[..., None, :]) ** 3
        )  # batch_shape, c,z
        valid_starting_waves = (
            (idx_z_source == level_idx)[..., None, :] * (Q0 < 1)
        ).any(axis=-1, keepdims=True)
        breaking_waves = (
            (Q0 >= 1) * (level_idx > idx_z_source)[..., None, :] * valid_starting_waves
        )

        breaking_idx = da.where(
            breaking_waves.any(axis=-1, keepdims=True),
            da.argmax(breaking_waves, axis=-1, keepdims=True),
            Q0.shape[-1] * self.damp_top - 1,
        )
        breaking_idx = da.where(breaking_idx < tir_idx, breaking_idx, -1)
        breaking = (breaking_idx == level_idx)[..., None, :]
        f0 = da.sum(breaking * rho_0[..., None, None] * spectrum[..., None], axis=-2)
        rho_half = da.concatenate(
            [rho[..., 0], np.sqrt(rho[..., 1:] * rho[..., :-1])], axis=-1
        )
        half_lvl_drag = f0 * eps[..., None] / (rho_half * dz)
        full_lvl_drag = da.concatenate(
            [
                0.5 * (half_lvl_drag[..., 1:] + half_lvl_drag[..., :-1]),
                half_lvl_drag[..., -1, None],
            ],
            axis=-1,
        )
        return full_lvl_drag.chunk(u.chunks)

    def gwd(self, u, N, z, rho):
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
        wave_mask = np.ones_like(self.c0, dtype=bool)

        ## Source level parameters
        rho_0 = rho_sources[0]
        u_0 = u_sources[0]
        source_spectrum = self.source_spectrum(self.c0, u_0)

        for i, (z_lvl, u_lvl, N_lvl, rho_lvl) in enumerate(
            zip(z_sources, u_sources, N_sources, rho_sources)
        ):
            intrinsic_freq = self.kwv * (self.c0 - u_lvl)
            # Total internal reflection
            H = -(z_lvl - z[idx_z_source + i - 1]) / np.log(
                rho_lvl / rho[idx_z_source + i - 1]
            )
            not_tir = np.abs(intrinsic_freq) < self.reflection_frequency(N_lvl, H)
            wave_mask = wave_mask & not_tir
            # Breaking condition
            if not wave_mask.any():
                break
            Q0 = (
                2
                * N_lvl
                * source_spectrum
                * rho_0
                / (rho_lvl * self.kwv * (self.c0 - u_lvl) ** 3)
            )

            if i > 0:
                # Above breaking level check for losses
                if i == len(z_sources) - 1 and self.damp_top:
                    breaking_waves = wave_mask
                else:
                    breaking_waves = wave_mask & (Q0 >= 1)
                dz = z_lvl - z_sources[i - 1]
                F0 = np.sum(rho_0 * source_spectrum[breaking_waves])
                rho_half = np.sqrt(rho_lvl * rho[idx_z_source + i - 1])
                X_half = F0 * self.intermittency(rho_0, u_0) / (rho_half * dz)
                drag[idx_z_source + i] = X_half
                drag[idx_z_source + i - 1] = 0.5 * (X_half + drag[idx_z_source + i - 1])

                # # Last level
                # breaking_waves = wave_mask
                # dz = z_lvl - z_sources[i-1]
                # F0 = np.sum(rho_0*self.source_spectrum(self.c0[breaking_waves],u_0))
                # rho_half = np.sqrt(rho_lvl*rho[idx_z_source + i-1])
                # X_half = F0*self.intermittency(rho_0,u_0)/(rho_half*dz)
                # drag[-1] = X_half
                # drag[-2] = drag[-2] * 0.5

            wave_mask = wave_mask & (Q0 < 1)
        return drag

    def gwd_momentum_flux(self, gwd, z, rho):
        uw = -1 * cumulative_trapezoid(rho[::-1] * gwd[::-1], x=z[::-1], initial=0)
        return uw

    def gwd_momentum_flux_ptv(self, gwd, z, rho):
        """
        Calculate the momentum flux due to gravity waves only
        """
        gwd_ptv = np.where(gwd > 0, gwd, 0)

        # Integrate from top to bottom
        # We are working on height levels here (integral is dz not dp/g )
        uw = (
            -1
            * cumulative_trapezoid(rho[::-1] * gwd_ptv[::-1], x=z[::-1], initial=0)[
                ::-1
            ]
        )
        return uw

    def gwd_momentum_flux_ntv(self, gwd, z, rho):
        """
        Calculate the momentum flux due to gravity waves only
        """
        drag_ntv = np.where(gwd < 0, gwd, 0)

        # Integrate from top to bottom
        # We are working on height levels here (integral is dz not dp/g )
        uw = (
            -1
            * cumulative_trapezoid(rho[::-1] * drag_ntv[::-1], x=z[::-1], initial=0)[
                ::-1
            ]
        )
        return uw
