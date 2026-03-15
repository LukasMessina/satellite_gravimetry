from dataclasses import dataclass
from helpers import wrap_rad
from tudatpy.astro import element_conversion
from scipy.integrate import quad
from scipy.optimize import brentq

import math


@dataclass
class OrbitalElements:
    """Class to hold the classical orbital elements of a satellite orbit."""
    a_km: float                 # semi-major axis            [km]
    e: float                    # eccentricity               [-]
    i_deg: float                # inclination                [deg]
    raan_deg: float             # RAAN Ω                     [deg]
    argp_deg: float             # argument of periapsis ω    [deg]
    M_deg: float                # mean anomaly at epoch      [deg]

    def get_along_track_shift(
        self, 
        separation_km: float,
        tol_abs: float = 1e-13,
        tol_rel: float = 1e-13,
        ) -> 'OrbitalElements':
        """Get the orbital elements of the targeter given a separation from the chaser.

        Args:
            separation_km (float): Along-track separation between chaser and targeter [km].
        """
        if separation_km == 0.0:
            return self
        if self.a_km <= 0:
            raise ValueError("Semi-major axis must be positive.")


        separation_m = separation_km * 1e3
        M_rad = math.radians(self.M_deg)
        nu0_rad = element_conversion.mean_to_true_anomaly(mean_anomaly=M_rad, eccentricity=self.e)
        p_m = self.a_km * 1e3 * (1.0 - self.e**2)

        # Integrand to compute arc length in true anomaly
        def integrand(nu: float) -> float:
            denom = 1.0 + self.e * math.cos(nu)
            radius_m = p_m / denom
            dr_dnu = (p_m * self.e * math.sin(nu)) / (denom**2)
            return math.sqrt(radius_m**2 + dr_dnu**2)
        
        def arc_length(nu1: float) -> float:
            """Compute the arc length between true anomalies."""
            arc_length, _ = quad(integrand, nu0_rad, nu1, epsabs=tol_abs, epsrel=tol_rel)
            return arc_length
        
        # Root function
        def root_function(dnu_rad: float) -> float:
            return arc_length(nu1 = nu0_rad + dnu_rad) - separation_m

        # Root bracketing strategy (single-revolution, smallest-magnitude solution)
        # Use local radius as initial guess; expand until bracket.
        initial_radius = p_m / (1.0 + self.e * math.cos(nu0_rad))
        dnu_guess = separation_m / initial_radius # good approximation for near-circular, small separations
        # starting interval around inital guess
        lower_bound = dnu_guess * 0.5
        upper_bound = dnu_guess * 1.5
        
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        # Expand bracket until sign change or until we hit +/-2π (avoid multi-revolution ambiguity)
        func_value_lower_bound = root_function(lower_bound)
        func_value_upper_bound = root_function(upper_bound)
        max_abs = 2.0 * math.pi - 1e-9
        expand = 1.5
        iter = 0
        while func_value_lower_bound * func_value_upper_bound > 0 and (abs(lower_bound) < max_abs and abs(upper_bound) < max_abs) and iter < 50:
            lower_bound *= expand
            upper_bound *= expand
            # clamp
            lower_bound = max(-max_abs, min(max_abs, lower_bound))
            upper_bound = max(-max_abs, min(max_abs, upper_bound))
            if lower_bound > upper_bound:
                lower_bound, upper_bound = upper_bound, lower_bound
            func_value_lower_bound = root_function(lower_bound)
            func_value_upper_bound = root_function(upper_bound)
            iter += 1

        if func_value_lower_bound * func_value_upper_bound > 0:
            raise RuntimeError("Could not bracket Δν for the requested separation within one revolution.")

        dnu = brentq(root_function, lower_bound, upper_bound, xtol=1e-13, rtol=1e-13, maxiter=200)

        # Update true anomaly, then convert back to mean anomaly using Tudat
        nu1 = wrap_rad(nu0_rad + dnu)
        final_eccentric_anomaly = element_conversion.true_to_eccentric_anomaly(true_anomaly=nu1, eccentricity=self.e)
        final_mean_anomaly = element_conversion.eccentric_to_mean_anomaly(eccentric_anomaly=final_eccentric_anomaly, eccentricity=self.e)

        return OrbitalElements(
            a_km=self.a_km,
            e=self.e,
            i_deg=self.i_deg,
            raan_deg=self.raan_deg,
            argp_deg=self.argp_deg,
            M_deg=math.degrees(final_mean_anomaly),
        )



