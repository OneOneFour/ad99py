from scipy.integrate import cumulative_trapezoid
from typing import Literal, NamedTuple, Optional, Union,Callable
from numbers import Number
import numpy as np
from numpy.typing import ArrayLike, NDArray

## CONSTANTS


## level struct
class Level(NamedTuple):
    level: int
    z: float
    u: float
    N: float
    rho: float
    dz: float
    H: float
    Q0: ArrayLike
    breaking_waves: ArrayLike
    reflected_waves: ArrayLike
    top: bool = False
    source: bool = False


## Class implementation


class AlexanderDunkerton1999:
    def __init__(
        self,
        source:Callable[[Number,Number],Number]=None,
        source_level_height: Number = 9e3,
        Fs0: Number = 0.004,
        damp_level_height: Optional[Number] = None,
        cmax: Number = 99.6,
        dc: Number = 1.2,
        base_wavelength: Number = 300e3,
        force_intermittency: Optional[Number] = None,
        use_intrinsic_c: Union[Number, Literal["never", "always"]] = "always",
        no_alpha: bool = False,
        exclude_unbroken: bool = False,
        cw: float = 35,
        Bm:float = 0.4,
    ):
        """
        Initialize an AlexanderDunkerton1999 Non-orographic drag parameterization instance.
        Following Alexander & Dunkerton 1999 (Journal of the Atmospheric Sciences)

        Parameterization Parameters:
        Bt: Total Equatorial GW Momentum Flux (m^2/s^2)

        Grid Parameters:
        pfull - Full pressure levels (hPa) [N]
        rho - Density on full pressure levels (kg/m^3) [N]\

        """
        self.dc = dc
        self.cmax = cmax
        if source is None:
            from warnings import warn
            from .sources import make_source_spectrum,gaussian_source
            warn(f"`source` is not set, using default Gaussian source spectrum, with `cw={cw}` and `Bm={Bm}`.")
            source = make_source_spectrum(gaussian_source, cw,Bm)
        self.source = source
        self.dc = dc
        self.Fs0 = Fs0
        self.use_intrinsic_c = use_intrinsic_c
        self.source_level_height = source_level_height
        self.damp_level_height = damp_level_height
        ## Could be increased to consider higher wave vectors but for simplicity nk=1
        self.base_wavelength = base_wavelength
        self.force_intermittency = force_intermittency
        self.exclude_unbroken = exclude_unbroken  # Are unbroken waves include in the source spectrum or not? This only matters if the no top breaking is considered.
        self.no_alpha = no_alpha  # if set the bouyancy frequency is the TIR frequency
        self.c0 = np.arange(-self.cmax, self.cmax + self.dc, self.dc)
        self.kwv = 2 * np.pi / self.base_wavelength

    @property
    def exclude_topwaves(self) -> bool:
        return self.exclude_unbroken and not self.damp_level_height

    def __repr__(self) -> str:
        return (
            f"AlexanderDunkerton1999(Fs0={self.Fs0},height={self.source_level_height},"
            f" λ={self.base_wavelength}, damping_level={self.damp_level_height})"
        )

    def get_source_level(self, z: NDArray, lat: Optional[Number] = None) -> int:
        return int(
            np.argmin(np.abs(z -  self.get_source_height(lat=lat)))
        )

    def get_source_height(self,lat=None):
        if lat is None:
            lat = 0.0 # Assume at equator
        return self.source_level_height*np.cos(np.deg2rad(lat))

    def propagate_upwards(
        self,
        z: NDArray,
        u: NDArray,
        N: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
        c: Optional[NDArray] = None,
    ):
        if c is None:
            c = self.c0
        source_level = self.get_source_level(z, lat=lat)
        us = u[source_level:]
        Ns = N[source_level:]
        zs = z[source_level:]
        rhos = rho[source_level:]
        wave_mask = np.ones_like(c, dtype=bool)
        rho_0 = rhos[0]
        u_0 = us[0]
        spectrum = self.source_spectrum(c, u_0, lat=lat)
        for i, (z_lvl, u_lvl, N_lvl, rho_lvl) in enumerate(zip(zs, us, Ns, rhos)):
            level = source_level + i
            intrinsic_freq = self.kwv * (c - u_lvl)
            dz = z_lvl - z[level - 1]
            H = -(dz) / np.log(rho_lvl / rho[level - 1])
            # Total internal reflection
            tir_candidates = np.abs(intrinsic_freq) >= self.reflection_frequency(
                N_lvl, H
            )
            reflected_waves = tir_candidates & wave_mask
            wave_mask = wave_mask & (~tir_candidates)

            # Breaking condition
            Q0 = 2 * N_lvl * spectrum * rho_0 / (rho_lvl * self.kwv * (c - u_lvl) ** 3)
            strongshear = ((c - u_0) * (c - u_lvl)) <= 0
            breaking_waves = wave_mask & ((Q0 >= 1) | strongshear)
            yield Level(
                source=(i == 0),
                level=level,
                z=z_lvl,
                u=u_lvl,
                N=N_lvl,
                rho=rho_lvl,
                dz=dz,
                H=H,
                Q0=Q0,
                reflected_waves=reflected_waves,
                breaking_waves=breaking_waves,
            )
            wave_mask = wave_mask & ~breaking_waves

        # Use MiMA convention for the top level
        z_top = 2 * zs[-1] - zs[-2]
        u_top = 2 * us[-1] - us[-2]
        rho_top = 2 * rhos[-1] - rhos[-2]
        N_top = Ns[-1]
        dz_top = z_top - zs[-1]
        H_top = -(dz_top) / np.log(rhos[-1] / rhos[-2])
        tir_candidates = np.abs(self.kwv * (c - u_top)) >= self.reflection_frequency(
            N_top, H_top
        )
        reflected_waves = tir_candidates & wave_mask
        wave_mask = wave_mask & (~tir_candidates)

        yield Level(
            top=True,
            level=level + 1,
            z=z_top,
            u=u_top,
            N=N_top,
            rho=rho_top,
            dz=dz_top,
            H=H_top,
            Q0=np.ones_like(c) * np.nan,
            reflected_waves=reflected_waves,
            breaking_waves=wave_mask,
        )

    def iterate_damp_levels(self, z: NDArray, u: NDArray, N: NDArray, rho: NDArray):
        damp_level = self.get_damp_level(z)
        if damp_level is None:
            return
        z_damp = z[damp_level:]
        N_damp = N[damp_level:]
        rho_damp = rho[damp_level:]
        u_damp = u[damp_level:]
        for i, (z_lvl, u_lvl, N_lvl, rho_lvl) in enumerate(
            zip(z_damp, u_damp, N_damp, rho_damp)
        ):
            level = damp_level + i
            dz = z_lvl - z[level - 1]
            H = -(dz) / np.log(rho_lvl / rho[level - 1])
            yield Level(
                level=level,
                z=z_lvl,
                u=u_lvl,
                N=N_lvl,
                rho=rho_lvl,
                dz=dz,
                H=H,
                Q0=np.ones_like(self.c0) * np.nan,
                reflected_waves=np.zeros_like(self.c0, dtype=bool),
                breaking_waves=np.zeros_like(self.c0, dtype=bool),
            ), len(z_damp)

    def get_damp_level(self, z: NDArray) -> Optional[int]:
        if self.damp_level_height is None:
            return None
        else:
            damp_height = int(np.argmax(z > self.damp_level_height))
            return damp_height

    def source_spectrum(self, c: ArrayLike, u: ArrayLike, lat: Optional[Number] = None):
        """
        Gaussian source spectrum of GW momentum flux.
        Intrinsic phase speed is used by default (that is spectrum is centered on the source spectrum).
        This yields a spectrum that has no net momentum flux.

        Alternative is to center the spectrum on zero ground relative speed.
        This will yield a spectrum that has a netf momentum flux.

        $c_w$ determines the half width of the spectrum.
        $c_max$ determines the maximum phase speed of the spectrum.
        $delta_c$ determines the grid spacing of the spectrum.
        """
        

        if lat is None:
            lat = float("nan")
        match self.use_intrinsic_c:
            case "never":
                c0 = 0.0
            case "always":
                c0 = u
            case _:
                c0 = 0.0 if np.abs(lat) >= self.use_intrinsic_c else u
        return np.sign(c-u)*self.source(c,c0)
        # return (
        #     np.sign(c - u)
        #     * self.Bm
        #     * np.exp(-np.log(2) * ((c - c0) / self.cw) ** 2)
        #     * (~np.isclose(c - u, 0.0))
        # )

    def intermittency(
        self,
        rho_source: Number,
        u: Optional[ArrayLike] = None,
        lat: Optional[Number] = None,
    ):
        if self.force_intermittency:
            return self.force_intermittency
        return (self.Fs0 * self.dc) / (
            rho_source
            * np.sum(np.abs(self.source_spectrum(self.c0, u, lat=lat) * self.dc))
        )

    def reflection_frequency(self, N: ArrayLike, H: ArrayLike):
        if self.no_alpha:
            return N
        alpha = 1 / (2 * H)
        return np.sqrt(N**2 * self.kwv**2 / (self.kwv**2 + alpha * alpha))

    def inspect_monochromatic(
        self,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        c: NDArray,
        lat: Optional[Number] = None,
    ):
        Q0 = np.zeros_like(u)
        omega = np.zeros_like(u)  # intrinsic frequency
        ref_freq = np.zeros_like(u)
        for level in self.propagate_upwards(z, u, N, rho, lat=lat, c=c):
            if level.top:
                ## Ignore residual
                break
            Q0[level.level] = level.Q0
            omega[level.level] = self.kwv * (c - level.u)
            ref_freq[level.level] = self.reflection_frequency(level.N, level.H)
        return Q0, omega, ref_freq

    def get_breaking_levels(
        self,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
    ):

        tir_levels = np.ones_like(self.c0, dtype=int) * -1
        breaking_levels = np.ones_like(self.c0, dtype=int) * -1

        for level in self.propagate_upwards(z, u, N, rho, lat=lat):
            if level.top:
                break
            tir_levels[level.reflected_waves] = level.level
            breaking_levels[level.breaking_waves] = level.level

        return tir_levels, breaking_levels

    def momentum_flux_abs(
        self,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
    ):
        momentum_flux_abs = np.zeros_like(u)
        source_level = self.get_source_level(z, lat=lat)
        u_source = u[source_level]
        rho_source = rho[source_level]
        eps = self.intermittency(rho_source, u_source, lat=lat)
        source_spectrum = (
            self.source_spectrum(self.c0, u_source, lat=lat) * rho_source * eps
        )
        momentum_flux_abs[:] = np.sum(np.abs(source_spectrum))
        for level in self.propagate_upwards(z, u, N, rho, lat=lat):
            if level.source:
                momentum_flux_abs[: level.level + 1] -= np.sum(
                    np.abs(
                        source_spectrum[
                            (level.breaking_waves) | (level.reflected_waves)
                        ]
                    )
                )
            elif level.top:
                momentum_flux_abs[: level.level + 1] -= np.sum(
                    np.abs(source_spectrum[level.reflected_waves])
                )
                if self.damp_level_height:
                    # deposit residual momentum flux amongst top levels
                    total_top_momentum_flux = np.sum(
                        np.abs(source_spectrum[level.breaking_waves])
                    )
                    for damplevel, n_levels in self.iterate_damp_levels(z, u, N, rho):
                        momentum_flux_abs[damplevel.level :] -= (
                            total_top_momentum_flux / n_levels
                        )
                elif self.exclude_unbroken:
                    total_top_momentum_flux = np.sum(
                        np.abs(source_spectrum[level.breaking_waves])
                    )
                    momentum_flux_abs[: level.level + 1] -= total_top_momentum_flux
            else:
                momentum_flux_abs[: level.level + 1] -= np.sum(
                    np.abs(source_spectrum[level.reflected_waves])
                )
                momentum_flux_abs[level.level] = momentum_flux_abs[
                    level.level - 1
                ] - np.sum(np.abs(source_spectrum[(level.breaking_waves)]))
        return np.where(momentum_flux_abs < 0, 0, momentum_flux_abs)

    def momentum_flux_neg_ptv(
        self,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
    ):
        """
        Alternative accounting method, without a drag integral to diagnose momentum flux.
        Removes compensating flux.
        """
        momentum_flux_neg = np.zeros_like(u)
        momentum_flux_ptv = np.zeros_like(u)
        source_level = self.get_source_level(z, lat=lat)
        u_source = u[source_level]
        rho_source = rho[source_level]
        eps = self.intermittency(rho_source, u_source, lat=lat)
        source_spectrum = (
            self.source_spectrum(self.c0, u_source, lat=lat) * rho_source * eps
        )
        momentum_flux_ptv[:] = np.sum(source_spectrum[source_spectrum > 0])
        momentum_flux_neg[:] = np.sum(source_spectrum[source_spectrum < 0])
        for level in self.propagate_upwards(z, u, N, rho, lat=lat):
            if level.source:
                unstable_at_source = source_spectrum[
                    level.breaking_waves | level.reflected_waves
                ]
                momentum_flux_neg[:] -= np.sum(
                    unstable_at_source[unstable_at_source < 0]
                )
                momentum_flux_ptv[:] -= np.sum(
                    unstable_at_source[unstable_at_source > 0]
                )

            else:
                reflected = source_spectrum[level.reflected_waves]
                momentum_flux_neg[: level.level + 1] -= np.sum(reflected[reflected < 0])
                momentum_flux_ptv[: level.level + 1] -= np.sum(reflected[reflected > 0])
                if level.top:
                    top_ptv = np.sum(
                        source_spectrum[
                            level.breaking_waves & ((self.c0 - u_source) > 0)
                        ]
                    )
                    top_ntv = np.sum(
                        source_spectrum[
                            level.breaking_waves & ((self.c0 - u_source) < 0)
                        ]
                    )
                    if self.damp_level_height:
                        for damplevel, num_levels in self.iterate_damp_levels(
                            z, u, N, rho
                        ):
                            momentum_flux_neg[damplevel.level :] -= top_ntv / num_levels
                            momentum_flux_ptv[damplevel.level :] -= top_ptv / num_levels
                    elif self.exclude_unbroken:
                        momentum_flux_neg[: level.level + 1] -= top_ntv
                        momentum_flux_ptv[: level.level + 1] -= top_ptv
                else:
                    breaking = source_spectrum[level.breaking_waves]
                    momentum_flux_neg[level.level] = momentum_flux_neg[
                        level.level - 1
                    ] - np.sum(breaking[breaking < 0])
                    momentum_flux_ptv[level.level] = momentum_flux_ptv[
                        level.level - 1
                    ] - np.sum(breaking[breaking > 0])

        return np.where(momentum_flux_neg < 0, momentum_flux_neg, 0), np.where(
            momentum_flux_ptv > 0, momentum_flux_ptv, 0
        )

    def gwd(
        self,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
    ):
        """
        Input:
        u - zonal wind profile (m/s) at each height level [...,N]
        N - Buoyancy frequency (s^-1) at each height level [...,N]

        Output:
        GWD - Non orographic Gravity wave drag at each height level [...,N]

        """
        drag = np.zeros_like(u)
        idx_source = self.get_source_level(z, lat=lat)
        u_source = u[idx_source]
        rho_source = rho[idx_source]
        source_spectrum = self.source_spectrum(self.c0, u_source, lat=lat)
        eps = self.intermittency(rho_source, u_source, lat=lat)
        for level in self.propagate_upwards(z, u, N, rho, lat=lat):
            if not level.top and not level.source:
                f0 = np.sum(source_spectrum[level.breaking_waves] * rho_source)
                rho_half = np.sqrt(rho[level.level - 1] * level.rho)
                drag_half_lvl = f0 * eps / (rho_half * level.dz)
                drag[level.level] = drag_half_lvl
                drag[level.level - 1] = 0.5 * (drag_half_lvl + drag[level.level - 1])

            if level.top:
                drag[-1] = drag[-1] / 2
                if self.damp_level_height:
                    # deposit residual momentum flux amongst top levels
                    total_top_momentum_flux = np.sum(
                        source_spectrum[level.breaking_waves]
                    )
                    for damplevel, n_levels in self.iterate_damp_levels(z, u, N, rho):
                        drag[damplevel.level] += (
                            eps
                            * rho_source
                            * total_top_momentum_flux
                            / (n_levels * level.rho * level.dz)
                        )
        return drag

    def gwd_net_momentum_flux_downwards(self, gwd: NDArray, z: NDArray, rho: NDArray):
        uw = -1 * cumulative_trapezoid(
            rho[::-1] * gwd[::-1], x=z[::-1], initial=0, axis=-1
        )
        return uw

    def gwd_abs_momentum_flux_downwards(self, gwd: NDArray, z: NDArray, rho: NDArray):
        """
        Calculate the absolute momentum flux due to gravity waves
        """
        uw = -1 * cumulative_trapezoid(
            rho[::-1] * np.abs(gwd[::-1]), x=z[::-1], initial=0, axis=-1
        )
        return uw

    def gwd_ptv_momentum_flux_downwards(self, gwd: NDArray, z: NDArray, rho: NDArray):
        """
        Calculate the momentum flux due to gravity waves only
        """
        gwd_ptv = np.where(gwd > 0, gwd, 0)

        # Integrate from top to bottom
        # We are working on height levels here (integral is dz not dp/g )
        uw = (
            -1
            * cumulative_trapezoid(
                rho[::-1] * gwd_ptv[::-1], x=z[::-1], initial=0, axis=-1
            )[::-1]
        )
        return uw

    def gwd_ntv_momentum_flux_downwards(self, gwd: NDArray, z: NDArray, rho: NDArray):
        """
        Calculate the momentum flux due to gravity waves only
        """
        drag_ntv = np.where(gwd < 0, gwd, 0)

        # Integrate from top to bottom
        # We are working on height levels here (integral is dz not dp/g )
        uw = (
            -1
            * cumulative_trapezoid(
                rho[::-1] * drag_ntv[::-1], x=z[::-1], initial=0, axis=-1
            )[::-1]
        )
        return uw

    def filtered_source_spectrum(
        self,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
    ) -> NDArray:
        """
        Returns the actual source spectrum has same dimensions as c0
        """
        idx_source = self.get_source_level(z, lat=lat)
        u_source = u[idx_source]
        rho_source = rho[idx_source]

        source_spectrum = self.source_spectrum(self.c0, u_source, lat=lat)
        eps = self.intermittency(rho_source, u_source, lat=lat)
        for level in self.propagate_upwards(z, u, N, rho, lat=lat):
            if level.source:
                source_spectrum[level.breaking_waves] = 0.0
            if level.reflected_waves.any():
                source_spectrum[level.reflected_waves] = 0.0
            if level.top and self.exclude_unbroken and not self.damp_level_height:
                source_spectrum[level.breaking_waves] = 0.0

        return source_spectrum * eps * rho_source

    def gwd_net_momentum_flux_upwards(
        self,
        gwd: NDArray,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
        source_spectrum: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Calculate the net momentum flux due to gravity waves.
        """
        # Integrate from top to bottom
        # We are working on height levels here (integral is dz not dp/g )
        if source_spectrum is None:
            filtered_source_spectrum = self.filtered_source_spectrum(u, N, z, rho, lat)
        else:
            filtered_source_spectrum = source_spectrum
        net_source_spectrum = filtered_source_spectrum.sum()

        uw = net_source_spectrum - 1 * cumulative_trapezoid(rho * gwd, x=z, initial=0)
        return uw

    def gwd_abs_momentum_flux_upwards(
        self,
        gwd: NDArray,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
        source_spectrum: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Calculate the absolute momentum flux due to gravity waves
        """
        if source_spectrum is None:
            filtered_source_spectrum = self.filtered_source_spectrum(u, N, z, rho, lat)
        else:
            filtered_source_spectrum = source_spectrum
        net_source_spectrum = np.abs(filtered_source_spectrum).sum()

        uw = net_source_spectrum - 1 * cumulative_trapezoid(
            rho * np.abs(gwd), x=z, initial=0
        )
        return np.where(uw > 0, uw, 0)

    def gwd_ptv_momentum_flux_upwards(
        self,
        gwd: NDArray,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
        source_spectrum: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Calculate the momentum flux due to gravity waves only
        """
        if source_spectrum is None:
            filtered_source_spectrum = self.filtered_source_spectrum(u, N, z, rho, lat)
        else:
            filtered_source_spectrum = source_spectrum
        net_source_spectrum = filtered_source_spectrum[
            filtered_source_spectrum > 0
        ].sum()

        gwd_ptv = np.where(gwd > 0, gwd, 0)

        # Integrate from top to bottom
        # We are working on height levels here (integral is dz not dp/g )
        uw = net_source_spectrum - 1 * cumulative_trapezoid(
            rho * gwd_ptv, x=z, initial=0
        )
        return np.where(uw > 0, uw, 0)

    def gwd_ntv_momentum_flux_upwards(
        self,
        gwd: NDArray,
        u: NDArray,
        N: NDArray,
        z: NDArray,
        rho: NDArray,
        lat: Optional[Number] = None,
        source_spectrum: Optional[NDArray] = None,
    ):
        """
        Calculate the momentum flux due to gravity waves only
        """
        if source_spectrum is None:
            filtered_source_spectrum = self.filtered_source_spectrum(u, N, z, rho, lat)
        else:
            filtered_source_spectrum = source_spectrum
        net_source_spectrum = filtered_source_spectrum[
            filtered_source_spectrum < 0
        ].sum()

        gwd_ntv = np.where(gwd < 0, gwd, 0)

        # Integrate from top to bottom
        # We are working on height levels here (integral is dz not dp/g )
        uw = net_source_spectrum - 1 * cumulative_trapezoid(
            rho * gwd_ntv, x=z, initial=0
        )
        return np.where(uw < 0, uw, 0)
