from typing import Optional
import dask.array as da
import numpy as np
from ad99 import AlexanderDunkerton1999


def dask_take_along_axis(data, idx):
    return np.take_along_axis(data, idx, axis=-1).squeeze(axis=-1)


class AlexanderDunkerton1999Vectorized(AlexanderDunkerton1999):
    def get_source_level(self, z, lat=None):
        if lat is None:
            lat = np.zeros_like(z)
        return np.argmin(
            np.abs(z - self.source_level_height * np.cos(np.deg2rad(lat))[...,None]),
            axis=-1,
            keepdims=True,
        )

    def intermittency(self, rho_source, u=None, lat=None):
        if self.force_intermittency:
            return self.force_intermittency
        return (self.Fs0 * self.dc) / (
            # Technically this is a stricter implementation of the AD99 definition of eps
            rho_source
            * np.sum(
                np.abs(self.source_spectrum(self.c0, u, lat=lat)) * self.dc, axis=-1
            )
        )

    def source_spectrum(self, c, u, lat=None):
        u = u[..., None]
        if lat is None:
            c0 = u
        else:
            if self.use_intrinsic_c == "always":
                c0 = u
            elif self.use_intrinsic_c == "never":
                c0 = 0.0
            else:
                c0 = np.where(np.abs(lat) > self.use_intrinsic_c, 0.0, u)
        return (
            np.sign(c - u)
            * self.Bm
            * np.exp(-np.log(2) * ((c - c0) / self.cw) ** 2)
            * (~np.isclose(c - u, 0.0))
        )

    def get_source_variables(self, z, u, N, rho, lat=None):
        source_levels = self.get_source_level(z, lat)
        rho_0 = np.take_along_axis(rho, source_levels, axis=-1)
        u_0 = np.take_along_axis(u, source_levels, axis=-1)
        return rho_0, u_0

    def get_vertical_scales(self, z, rho):
        # calculate spacing constants
        dz = z[..., 1:] - z[..., :-1]
        H = -(dz / np.log(rho[..., 1:] / rho[..., :-1]))
        dz = np.concatenate([dz[..., 0, None], dz], axis=-1)
        H = np.concatenate([H[..., 0, None], H], axis=-1)
        return dz, H

    def propagate_upwards(
        self,
        z: np.ndarray,
        u: np.ndarray,
        N: np.ndarray,
        rho: np.ndarray,
        lat: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
    ):
        if c is None:
            c = self.c0
        else:
            c = np.asarray(c)

        level_idx = np.broadcast_to(np.arange(z.shape[-1], dtype=int), z.shape)
        source_levels = self.get_source_level(z, lat)
        rho_0, u_0 = self.get_source_variables(z, u, N, rho, lat=lat)
        spectrum = self.source_spectrum(c, u_0, lat=lat)

        # calculate spacing constants
        dz, H = self.get_vertical_scales(z, rho)
        reflection_frequency = self.reflection_frequency(N, H)
        intrinsic_freq = self.kwv * (c[..., :, None] - u[..., None, :])
        tir = (np.abs(intrinsic_freq) >= reflection_frequency[..., None, :]) * (
            (level_idx >= source_levels)[..., None, :]
        )

        tir_idx = np.where(
            tir.any(axis=-1, keepdims=True),
            np.argmax(tir, axis=-1, keepdims=True),
            tir.shape[-1] + 1,
        )

        Q0 = (2 * N[..., None, :] * spectrum[..., :, None] * rho_0[..., None, None]) / (
            rho[..., None, :]
            * self.kwv
            * (self.c0[..., :, None] - u[..., None, :]) ** 3
        )  # batch_shape, c,z

        signchange = (self.c0[..., :, None] - u[..., None, :]) * (
            self.c0[..., :, None] - u_0[..., None, None]
        ) <= 0
        breaking_waves = ((Q0 >= 1) | (signchange)) & (level_idx >= source_levels)[
            ..., None, :
        ]
        breaking_idx = np.where(
            breaking_waves.any(axis=-1, keepdims=True),
            np.argmax(breaking_waves, axis=-1, keepdims=True),
            breaking_waves.shape[-1],
        )
        true_breaking_idx = np.where(breaking_idx < tir_idx, breaking_idx, -1)
        true_reflecting_idx = np.where(tir_idx <= breaking_idx, tir_idx, -1)
        breaking = true_breaking_idx == level_idx[..., None, :]
        reflecting = true_reflecting_idx == level_idx[..., None, :]
        topwaves = breaking_idx == z.shape[-1]
        return breaking, reflecting, topwaves

    def gwd(
        self,
        z: np.ndarray,
        u: np.ndarray,
        N: np.ndarray,
        rho: np.ndarray,
        lat: Optional[np.ndarray] = None,
    ):
        """
        vectorized dask friendly implementation
        """
        rho_0, u_0 = self.get_source_variables(z, u, N, rho, lat=lat)
        dz, _ = self.get_vertical_scales(z, rho)
        spectrum = self.source_spectrum(self.c0, u_0, lat=lat)
        eps = self.intermittency(rho_0, u_0, lat=lat)

        breaking, reflecting, topwaves = self.propagate_upwards(z, u, N, rho, lat=lat)

        F0 = np.sum(breaking * rho_0[..., None, None] * spectrum[..., None], axis=-2)
        rho_half = np.concatenate(
            [rho_0[..., None], np.sqrt(rho[..., 1:] * rho[..., :-1])], axis=-1
        )
        half_lvl_drag = F0 * eps[..., None] / (rho_half * dz)
        full_lvl_drag = np.concatenate(
            [
                0.5 * (half_lvl_drag[..., 1:] + half_lvl_drag[..., :-1]),
                0.5 * half_lvl_drag[..., -1][..., None],
            ],
            axis=-1,
        )
        return full_lvl_drag




class AlexanderDunkerton1999Dask(AlexanderDunkerton1999):

    def get_source_level(self, z, lat=None):
        if lat is None:
            lat = da.zeros_like(z)
        return da.argmin(
            da.abs(z - self.source_level_height * da.cos(da.deg2rad(lat))[...,None]),
            axis=-1,
            keepdims=True,
        )

    def intermittency(self, rho_source, u=None, lat=None):
        if self.force_intermittency:
            return self.force_intermittency
        return (self.Fs0 * self.dc) / (
            # Technically this is a stricter implementation of the AD99 definition of eps
            rho_source
            * da.sum(
                da.abs(self.source_spectrum(self.c0, u, lat=lat)) * self.dc, axis=-1
            )
        )

    def source_spectrum(self, c, u, lat=None):
        u = u[..., None]
        if lat is None:
            c0 = u
        else:

            if self.use_intrinsic_c == "always":
                c0 = u
            elif self.use_intrinsic_c == "never":
                c0 = 0.0
            else:
                c0 = da.where(da.abs(lat[...,None]) > self.use_intrinsic_c, 0.0, u)
        return (
            da.sign(c - u)
            * self.Bm
            * da.exp(-da.log(2) * ((c - c0) / self.cw) ** 2)
            * (~da.isclose(c - u, 0.0))
        )

    def get_source_variables(self, z, u, N, rho, lat=None):
        source_levels = self.get_source_level(z, lat)
        rho_0 = da.map_blocks(dask_take_along_axis,rho,source_levels,dtype=rho.dtype,drop_axis=-1)
        u_0 = da.map_blocks(dask_take_along_axis, u, source_levels, dtype=u.dtype, drop_axis=-1)
    
        return rho_0, u_0

    def get_vertical_scales(self, z, rho):
        # calculate spacing constants
        dz = z[..., 1:] - z[..., :-1]
        H = -(dz / da.log(rho[..., 1:] / rho[..., :-1]))
        dz = da.concatenate([dz[..., 0, None], dz], axis=-1)
        H = da.concatenate([H[..., 0, None], H], axis=-1)
        dz = dz.rechunk(z.chunks)
        H = H.rechunk(z.chunks)
        return dz, H

    def propagate_upwards(
        self,
        z: da.Array,
        u: da.Array,
        N: da.Array,
        rho: da.Array,
        lat: Optional[da.Array] = None,
        c: Optional[np.ndarray] = None,
    ):
        ### 
        # vectorized dask friendly implementation
        # Throughout this function dimensions are (*batch_size, n_spectral, n_levels)
        # Broacasting rules used for this throughout.
        # TODO: Remove duplicate broadcastings, though maybe dask already handles this? 
        ### 
        if c is None:
            c = self.c0
        else:
            c = np.asarray(c)

        level_idx = da.broadcast_to(
            da.arange(z.shape[-1], dtype=int), z.shape, chunks=z.chunks
        )
        source_levels = self.get_source_level(z, lat)
        rho_0, u_0 = self.get_source_variables(z, u, N, rho, lat=lat)
        spectrum = self.source_spectrum(c, u_0, lat=lat)

        # calculate spacing constants
        dz, H = self.get_vertical_scales(z, rho)
        reflection_frequency = self.reflection_frequency(N, H)
        intrinsic_freq = self.kwv * (c[..., :, None] - u[..., None, :])
        tir = (da.abs(intrinsic_freq) >= reflection_frequency[..., None, :]) * (
            (level_idx >= source_levels)[..., None, :]
        )

        tir_idx = da.where(
            tir.any(axis=-1, keepdims=True),
            da.argmax(tir, axis=-1, keepdims=True),
            tir.shape[-1] + 1,
        )

        Q0 = (2 * N[..., None, :] * spectrum[..., :, None] * rho_0[..., None, None]) / (
            rho[..., None, :]
            * self.kwv
            * (self.c0[..., :, None] - u[..., None, :]) ** 3
        )  # batch_shape, c,z

        signchange = (self.c0[..., :, None] - u[..., None, :]) * (
            self.c0[..., :, None] - u_0[..., None, None]
        ) <= 0
        breaking_waves = ((Q0 >= 1) | (signchange)) & (level_idx >= source_levels)[
            ..., None, :
        ]
        breaking_idx = da.where(
            breaking_waves.any(axis=-1, keepdims=True),
            da.argmax(breaking_waves, axis=-1, keepdims=True),
            breaking_waves.shape[-1],
        )
        true_breaking_idx = da.where(breaking_idx < tir_idx, breaking_idx, -1)
        true_reflecting_idx = da.where(tir_idx <= breaking_idx, tir_idx, -1)
        breaking = (true_breaking_idx == level_idx[..., None, :]) * (
            level_idx > source_levels
        )[..., None, :]
        reflecting = true_reflecting_idx == level_idx[..., None, :]
        topwaves = breaking_idx == z.shape[-1]
        source_unstable = true_breaking_idx == source_levels[..., None, :]
        return breaking, reflecting, topwaves, source_unstable

    def gwd(
        self,
        u: da.Array,
        N: da.Array,
        z: da.Array,
        rho: da.Array,
        lat: Optional[da.Array] = None,
    ):
        """
        vectorized dask friendly implementation
        """
        rho_0, u_0 = self.get_source_variables(z, u, N, rho, lat=lat)
        dz, _ = self.get_vertical_scales(z, rho)
        spectrum = self.source_spectrum(self.c0, u_0, lat=lat)
        eps = self.intermittency(rho_0, u_0, lat=lat)

        breaking, _, topwaves, _ = self.propagate_upwards(z, u, N, rho, lat=lat)

        F0 = da.sum(breaking * rho_0[..., None, None] * spectrum[..., None], axis=-2)
        rho_half = da.concatenate(
            [rho_0[..., None], np.sqrt(rho[..., 1:] * rho[..., :-1])], axis=-1
        )
        half_lvl_drag = F0 * eps[..., None] / (rho_half * dz)
        full_lvl_drag = da.concatenate(
            [
                0.5 * (half_lvl_drag[..., 1:] + half_lvl_drag[..., :-1]),
                0.5 * half_lvl_drag[..., -1][..., None],
            ],
            axis=-1,
        )
        full_lvl_drag = full_lvl_drag.rechunk(z.chunks)
        return full_lvl_drag

    def momentum_flux_abs(self, u, N, z, rho, lat=None):
        raise NotImplementedError("momentum_flux_abs is not implemented in this class; ")

    

    def momentum_flux_neg_ptv(self, u, N, z, rho, lat=None):
        rho_0, u_0 = self.get_source_variables(z, u, N, rho, lat=lat)
        u_0_brd = u_0[..., None]
        rho_0_brd = rho_0[..., None, None]
        spectrum = self.source_spectrum(self.c0, u_0, lat=lat)
        eps = self.intermittency(rho_0, u_0, lat=lat)
        ptv = ((self.c0 - u_0_brd) > 0)[..., :, None]
        ntv = ((self.c0 - u_0_brd) < 0)[..., :, None]
        breaking, reflecting, topwaves, source_unstable = self.propagate_upwards(
            z, u, N, rho, lat=lat
        )

        exclude = ~(source_unstable | reflecting.any(axis=-1, keepdims=True) | topwaves*self.exclude_topwaves)
        filtered = spectrum[..., None] * exclude * rho_0_brd * eps[..., None, None]
        breaking_ptv = breaking * ptv
        breaking_ntv = breaking * ntv

        dmomflux_ptv = (
            da.sum(breaking_ptv * rho_0_brd * spectrum[..., None], axis=-2)
            * eps[..., None]
        )
        dmomflux_ntv = (
            da.sum(breaking_ntv * rho_0_brd * spectrum[..., None], axis=-2)
            * eps[..., None]
        )

        momentum_flux_ptv = da.sum(filtered * ptv, axis=-2) - da.cumsum(
            dmomflux_ptv, axis=-1, method="blelloch"
        )
        momentum_flux_ntv = da.sum(filtered * ntv, axis=-2) - da.cumsum(
            dmomflux_ntv, axis=-1, method="blelloch"
        )
        return momentum_flux_ntv, momentum_flux_ptv
