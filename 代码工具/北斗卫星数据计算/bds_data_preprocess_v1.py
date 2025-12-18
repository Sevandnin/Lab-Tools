"""
BDS Data Preprocessing - Python Implementation
Converts MATLAB BDSData_preprocess.m and related functions to Python
"""

import numpy as np
import pandas as pd
from datetime import datetime
import math
import os


# ====================================================================================
# Helper Functions
# ====================================================================================

def str2doubleq(s):
    """
    Convert string to double, handling special cases like 'D' notation
    """
    if isinstance(s, str):
        s = s.strip().replace('D', 'e').replace('d', 'e')
        try:
            return float(s)
        except:
            return np.nan
    return s


def sind(angle_deg):
    """Sine function with degree input"""
    return np.sin(np.deg2rad(angle_deg))


def cosd(angle_deg):
    """Cosine function with degree input"""
    return np.cos(np.deg2rad(angle_deg))


def asind(value):
    """Arcsine function with degree output"""
    return np.rad2deg(np.arcsin(value))


def atan2d(y, x):
    """Atan2 function with degree output"""
    return np.rad2deg(np.arctan2(y, x))


# ====================================================================================
# Date and Time Conversion Functions
# ====================================================================================

def date_to_gps_time(year, month, day, hour_of_day):
    """
    Convert calendar date to GPS time
    
    Parameters:
    -----------
    year : int
        Year
    month : int
        Month (1-12)
    day : int
        Day of month
    hour_of_day : float
        Hour of day (decimal hours)
    
    Returns:
    --------
    tow : float
        GPS time of week (seconds)
    gps_week : int
        GPS week number
    """
    # Count leap years from 1980 to year-1
    number_of_leapyear = 0
    for y in range(1980, year):
        if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0):
            number_of_leapyear += 1
    
    # Calculate total days from GPS epoch
    total_day = 365 * (year - 1980) + number_of_leapyear - 5
    
    # Add days from months
    days_of_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for m in range(month - 1):
        total_day += days_of_month[m]
    
    # Add leap day if necessary
    if month > 2 and ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
        total_day += 1
    
    total_day += day - 1
    
    # Calculate GPS time of week and week number
    tow = (total_day % 7) * 3600 * 24 + hour_of_day * 3600
    gps_week = (total_day - (total_day % 7)) // 7
    
    return tow, gps_week


# ====================================================================================
# Coordinate Transformation Functions
# ====================================================================================

def xyz_to_blh(X, Y, Z):
    """
    Convert ECEF coordinates to geodetic coordinates (WGS84)
    
    Parameters:
    -----------
    X, Y, Z : float
        ECEF coordinates in meters
    
    Returns:
    --------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    h : float
        Height in meters
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis
    e2 = 0.00669438002290  # first eccentricity squared
    
    elat = 1e-12
    eht = 1e-5
    
    p = np.sqrt(X**2 + Y**2)
    lat = np.arctan2(Z, p / (1 - e2))
    h = 0
    dh = 1
    dlat = 1
    
    while dlat > elat or dh > eht:
        lat0 = lat
        h0 = h
        v = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p * np.cos(lat) + Z * np.sin(lat) - a**2 / v
        lat = np.arctan2(Z, p * (1 - e2 * v / (v + h)))
        dlat = abs(lat - lat0)
        dh = abs(h - h0)
    
    lon = np.arctan2(Y, X) * 180 / np.pi
    lat = lat * 180 / np.pi
    
    return lat, lon, h


# ====================================================================================
# Kepler Equation Solvers
# ====================================================================================

def kepler_initial(e, M):
    """
    Initial value for Kepler equation solution
    
    Parameters:
    -----------
    e : float
        Eccentricity
    M : float
        Mean anomaly (radians)
    
    Returns:
    --------
    E0 : float
        Initial eccentric anomaly
    """
    e_2 = e**2
    e_3 = e_2 * e
    cos_M = np.cos(M)
    
    E0 = M + (-0.5*e_3 + e + (e_2 + 1.5*e_3*cos_M)*cos_M) * np.sin(M)
    return E0


def kepler_eps1(e, M, x):
    """
    Iteration step for Kepler equation
    
    Parameters:
    -----------
    e : float
        Eccentricity
    M : float
        Mean anomaly
    x : float
        Current eccentric anomaly estimate
    
    Returns:
    --------
    eps1 : float
        Correction term
    """
    eps1 = (x - e * np.sin(x) - M) / (1 - e * np.cos(x))
    return eps1


def kepler_solve(e, M, tol=1e-10):
    """
    Solve Kepler equation: M = E - e*sin(E) for E
    
    Parameters:
    -----------
    e : float
        Eccentricity
    M : float
        Mean anomaly
    tol : float
        Tolerance for convergence
    
    Returns:
    --------
    E : float
        Eccentric anomaly
    """
    # Normalize M to [0, 2pi)
    Mnorm = M % (2*np.pi)

    # Initial guess: series expansion for small e, pi for high e
    if e < 0.8:
        E = kepler_initial(e, Mnorm)
    else:
        E = Mnorm if abs(Mnorm) < np.pi else np.pi

    max_iter = 50
    for _ in range(max_iter):
        f = E - e * np.sin(E) - Mnorm
        fprime = 1 - e * np.cos(E)
        # Newton step
        delta = -f / fprime
        E = E + delta
        if abs(delta) < tol:
            break
    else:
        print('Warning: Kepler equation failed to converge after', max_iter, 'iterations')

    return E


# ====================================================================================
# Tropospheric Delay Functions
# ====================================================================================

def zenith_delay(B, H, DOY):
    """
    Calculate Zenith Delay of troposphere using EGNOS Model
    
    Parameters:
    -----------
    B : float
        Latitude (degrees)
    H : float
        Height (meters)
    DOY : int
        Day of year
    
    Returns:
    --------
    ZHD : float
        Zenith Hydrostatic Delay (meters)
    ZWD : float
        Zenith Wet Delay (meters)
    """
    if B > 0:
        doymin = 28
    else:
        doymin = 211
    
    k = np.cos(2 * np.pi * (DOY - doymin) / 365.25)
    B_abs = abs(B)
    
    # Select parameters based on latitude
    if B_abs < 15:
        P = 1013.25 - 0*k
        beta = 6.3 - 0*k
        T = 299.65 - 0*k
        e = 26.31 - 0*k
        ranta = 2.77 - 0*k
    elif B_abs < 30:
        m = (B_abs - 15) / 15
        P = 1013.25 + m*(1017.25-1013.25) - (0+m*(-3.75-0))*k
        beta = 6.3 + m*(6.05-6.30) - (0+m*(0.25-0))*k
        T = 299.65 + m*(294.15-299.65) - (0+m*(7.00-0))*k
        e = 26.31 + m*(21.79-26.31) - (0+m*(8.85-0.0))*k
        ranta = 2.77 + m*(3.15-2.77) - (0+m*(0.33-0.0))*k
    elif B_abs < 45:
        m = (B_abs - 30) / 15
        P = 1017.25 + m*(1015.75-1017.25) - (-3.75+m*(-2.25+3.75))*k
        beta = 6.05 + m*(5.58-6.05) - (0.25+m*(0.32-0.25))*k
        T = 294.15 + m*(283.15-294.15) - (7.00+m*(11.00-7.00))*k
        e = 21.79 + m*(11.66-21.79) - (8.85+m*(7.24-8.85))*k
        ranta = 3.15 + m*(2.57-3.15) - (0.33+m*(0.46-0.33))*k
    elif B_abs < 60:
        m = (B_abs - 45) / 15
        P = 1015.75 + m*(1011.75-1015.75) - (-2.25+m*(-1.75+2.25))*k
        beta = 5.58 + m*(5.39-5.58) - (0.32+m*(0.81-0.32))*k
        T = 283.15 + m*(272.15-283.15) - (11.00+m*(15.00-11.00))*k
        e = 11.66 + m*(6.78-11.66) - (7.24+m*(5.36-7.24))*k
        ranta = 2.57 + m*(1.81-2.57) - (0.46+m*(0.74-0.46))*k
    elif B_abs < 75:
        m = (B_abs - 60) / 15
        P = 1011.75 + m*(1013.00-1011.75) - (-1.75+m*(-0.50+1.75))*k
        beta = 5.39 + m*(4.53-5.39) - (0.81+m*(0.62-0.81))*k
        T = 272.15 + m*(263.65-272.15) - (15.00+m*(14.50-15.00))*k
        e = 6.78 + m*(4.11-6.78) - (5.36+m*(3.39-5.36))*k
        ranta = 1.81 + m*(1.55-1.81) - (0.74+m*(0.33-0.74))*k
    else:
        P = 1013.00 + 0.5*k
        beta = 4.53 - 0.62*k
        T = 263.65 - 14.5*k
        e = 4.11 - 3.39*k
        ranta = 1.55 - 0.33*k
    
    # Calculate zenith delays
    ZHD = (77.604e-6) * 287.054 * P / 9.784
    ZWD = 0.382 * 287.054 / (9.784*(ranta+1) - beta/1000*287.054) * e / T
    ZHD = ZHD * (1 - beta/1000*H/T)**(9.80665/287.054/(beta/1000))
    ZWD = ZWD * (1 - beta/1000*H/T)**((ranta+1)*9.80665/287.054/(beta/1000)-1)
    
    return ZHD, ZWD


def neill_map(ZHD, ZWD, Ele, B, H, DOY):
    """
    Calculate Slant Total Delay using Neill Mapping Function
    
    Parameters:
    -----------
    ZHD : float
        Zenith Hydrostatic Delay (meters)
    ZWD : float
        Zenith Wet Delay (meters)
    Ele : float
        Satellite elevation (degrees)
    B : float
        Latitude (degrees)
    H : float
        Height (meters)
    DOY : int
        Day of year
    
    Returns:
    --------
    STD : float
        Slant Total Delay (meters)
    """
    # In MATLAB Neill_Map expects latitude in degrees and height in meters
    # Here B should be provided in degrees
    if B > 0:
        doymin = 28
    else:
        doymin = 211

    k = np.cos(2 * np.pi * (DOY - doymin) / 365.25)
    B_abs = abs(B)
    
    # Select parameters based on latitude
    if B_abs < 15:
        ah = 1.2769934e-3 - 0.0*k
        bh = 2.9153695e-3 - 0.0*k
        ch = 62.610505e-3 - 0.0*k
        aw = 5.8021897e-4
        bw = 1.4275268e-3
        cw = 4.3472961e-2
    elif B_abs < 30:
        m = (B_abs - 15) / 15
        ah = 1.2769934e-3 + m*(1.2683230-1.2769934)*1e-3 - (0.0+m*(1.2709626-0.0))*1e-5*k
        bh = 2.9153695e-3 + m*(2.9152299-2.9153695)*1e-3 - (0.0+m*(2.1414979-0.0))*1e-5*k
        ch = 62.610505e-3 + m*(62.837393-62.610505)*1e-3 - (0.0+m*(9.012840-0.0))*1e-5*k
        aw = 5.8021897e-4 + m*(5.6794847-5.8021897)*1e-4
        bw = 1.4275268e-3 + m*(1.5138625-1.4275268)*1e-3
        cw = 4.3472961e-2 + m*(4.6729510-4.3472961)*1e-2
    elif B_abs < 45:
        m = (B_abs - 30) / 15
        ah = 1.2683230e-3 + m*(1.2465397-1.2683230)*1e-3 - (1.2709626+m*(2.65236662-1.2709626))*1e-5*k
        bh = 2.9152299e-3 + m*(2.9288445-2.9152299)*1e-3 - (2.1414979+m*(3.0160779-2.1414979))*1e-5*k
        ch = 62.837393e-3 + m*(63.721774-62.837393)*1e-3 - (9.012840+m*(4.349703-9.012840))*1e-5*k
        aw = 5.6794847e-4 + m*(5.8118019-5.6794847)*1e-4
        bw = 1.5138625e-3 + m*(1.4572752-1.5138625)*1e-3
        cw = 4.6729510e-2 + m*(4.3908931-4.6729510)*1e-2
    elif B_abs < 60:
        m = (B_abs - 45) / 15
        ah = 1.2465397e-3 + m*(1.2196049-1.2465397)*1e-3 - (2.65236662+m*(3.4000452-2.65236662))*1e-5*k
        bh = 2.9288445e-3 + m*(2.9022565-2.9288445)*1e-3 - (3.0160779+m*(7.2562722-3.0160779))*1e-5*k
        ch = 63.721774e-3 + m*(63.824265-63.721774)*1e-3 - (4.349703+m*(84.795348-4.349703))*1e-5*k
        aw = 5.8118019e-4 + m*(5.9727542-5.8118019)*1e-4
        bw = 1.4572752e-3 + m*(1.5007428-1.4572752)*1e-3
        cw = 4.3908931e-2 + m*(4.4626982-4.3908931)*1e-2
    elif B_abs < 75:
        m = (B_abs - 60) / 15
        ah = 1.2196049e-3 + m*(1.2045996-1.2196049)*1e-3 - (3.4000452+m*(4.1202191-3.4000452))*1e-5*k
        bh = 2.9022565e-3 + m*(2.9024912-2.9022565)*1e-3 - (7.2562722+m*(11.723375-7.2562722))*1e-5*k
        ch = 63.824265e-3 + m*(64.258455-63.824265)*1e-3 - (84.795348+m*(170.37206-84.795348))*1e-5*k
        aw = 5.9727542e-4 + m*(6.1641693-5.9727542)*1e-4
        bw = 1.5007428e-3 + m*(1.7599082-1.5007428)*1e-3
        cw = 4.4626982e-2 + m*(5.4736038-4.4626982)*1e-2
    else:
        ah = 1.2045996e-3 - 4.1202191e-5*k
        bh = 2.9024912e-3 - 11.723375e-5*k
        ch = 64.258455e-3 - 170.37206e-5*k
        aw = 6.1641693e-4
        bw = 1.7599082e-3
        cw = 5.4736038e-2
    
    aht = 2.53e-5
    bht = 5.49e-3
    cht = 1.14e-3
    
    # Calculate mapping functions
    sin_ele = sind(Ele)
    MFh = ((1 + (ah/(1+bh/(1+ch)))) / (sin_ele + (ah/(sin_ele+bh/(sin_ele+ch)))) + 
           (1/sin_ele - (1+(aht/(1+bht/(1+cht))))/(sin_ele+(aht/(sin_ele+bht/(sin_ele+cht))))) * H/1000)
    MFw = (1 + (aw/(1+bw/(1+cw)))) / (sin_ele + (aw/(sin_ele+bw/(sin_ele+cw))))
    
    STD = ZHD * MFh + ZWD * MFw
    
    return STD


# ====================================================================================
# Ionospheric Delay Function
# ====================================================================================

def ionospheric_delay(gps_time, latitude, longitude, azimut, elevation, alpha, beta):
    """
    Calculate ionospheric delay using Klobuchar model
    
    Parameters:
    -----------
    gps_time : float
        GPS time of week (seconds)
    latitude : float
        Receiver latitude (degrees)
    longitude : float
        Receiver longitude (degrees)
    azimut : float
        Satellite azimuth (degrees)
    elevation : float
        Satellite elevation (degrees)
    alpha : array-like
        4 alpha coefficients for Klobuchar model
    beta : array-like
        4 beta coefficients for Klobuchar model
    
    Returns:
    --------
    T_iono : float
        Ionospheric delay (seconds)
    D_iono : float
        Ionospheric delay (meters)
    Std_iono : float
        Ionospheric standard deviation (meters)
    Var_iono : float
        Ionospheric variance (m²)
    """
    # Constants
    c = 299792458.0  # Speed of light [m/s]
    Re = 6378136.0   # Earth radius [m]
    hI = 350000.0    # Ionosphere altitude [m]
    
    # Convert to radians
    phi_u = np.deg2rad(latitude)
    lambda_u = np.deg2rad(longitude)
    A = np.deg2rad(azimut)
    E = np.deg2rad(elevation)
    
    # Psi - Earth central angle
    Psi = 0.0137 / (E/np.pi + 0.11) - 0.022
    
    # Pierce point latitude (semicircles)
    phi_i = phi_u/np.pi + Psi * np.cos(A)
    phi_i = min(phi_i, 0.416)
    phi_i = max(phi_i, -0.416)
    
    # Pierce point longitude (semicircles)
    lambda_i = lambda_u/np.pi + Psi * np.sin(A) / np.cos(phi_i * np.pi)
    
    # Local time at pierce point
    t = 4.32e4 * lambda_i + gps_time
    if t < 0:
        t += 86400
    elif t >= 86400:
        t -= 86400
    
    # Geomagnetic latitude (semicircles)
    phi_m = phi_i + 0.064 * np.cos((lambda_i - 1.617) * np.pi)
    
    # Obliquity factor
    F = 1.0 + 16.0 * (0.53 - E/np.pi)**3
    
    # Period (seconds)
    PER = beta[0] + beta[1]*phi_m + beta[2]*phi_m**2 + beta[3]*phi_m**3
    PER = max(PER, 72000)
    
    # Phase (radians)
    x = 2*np.pi * (t - 50400) / PER
    
    # Amplitude (seconds)
    AMP = alpha[0] + alpha[1]*phi_m + alpha[2]*phi_m**2 + alpha[3]*phi_m**3
    AMP = max(AMP, 0)
    
    # Ionospheric correction (seconds)
    T_iono = F * 5.0e-9
    if abs(x) < 1.57:
        T_iono = T_iono + F * AMP * (1 - x**2/2 + x**4/24)
    
    # Vertical deviation (meters)
    if abs(180*phi_m) <= 20:
        tau_vert = 9.0
    elif abs(180*phi_m) <= 55:
        tau_vert = 4.5
    else:
        tau_vert = 6.0
    
    # Obliquity factor
    Fpp = (1 - (Re/(Re+hI) * np.cos(E))**2)**(-0.5)
    
    # Variance
    Var_iono = max((c*T_iono/5)**2, (Fpp*tau_vert)**2)
    
    # Additional outputs
    D_iono = c * T_iono
    Std_iono = np.sqrt(Var_iono)
    
    return T_iono, D_iono, Std_iono, Var_iono


# ====================================================================================
# Satellite Position Calculation
# ====================================================================================

def cal_sv_pos_beidou(time_dict, obs_dict, nav_dict, obs_header):
    """
    Calculate BeiDou satellite position and clock bias
    
    Parameters:
    -----------
    time_dict : dict
        Time information (GPST, GPSWeek)
    obs_dict : dict
        Observation data
    nav_dict : dict
        Navigation message
    obs_header : dict
        Observation header information
    
    Returns:
    --------
    sv_pos : ndarray
        Satellite position [X, Y, Z] in meters
    sat_clk : float
        Satellite clock bias in seconds
    rate_clock : float
        Satellite clock drift
    Xsvel, Ysvel, Zsvel : float
        Satellite velocity components
    delta_tsv_L1pie : float
        Satellite clock frequency drift
    """
    # Constants
    c = 299792458.0  # Speed of light
    mu = 3.986004418e14  # Earth's gravitational parameter for BeiDou
    OMEGA_dot_e = 7.2921150e-5  # Earth's rotation rate for BeiDou
    
    # Get pseudorange
    pseudorange = None
    if 'C2I' in obs_dict and not np.isnan(obs_dict.get('C2I', np.nan)):
        pseudorange = obs_dict['C2I']
    elif 'C1I' in obs_dict and not np.isnan(obs_dict.get('C1I', np.nan)):
        pseudorange = obs_dict['C1I']
    
    if pseudorange is None:
        return None, None, None, None, None, None, None
    
    # Iteratively solve for satellite transmission time including satellite clock bias
    # Initial guess: neglect satellite clock
    sat_clk = 0.0
    max_iter = 6
    tol_clk = 1e-12
    t = time_dict['GPST'] - 14 - pseudorange / c  # initial BDS transmit time guess
    for _iter in range(max_iter):
        # Time from ephemeris reference epoch
        t_k = t - nav_dict['toe']
        if t_k > 302400:
            t_k = t_k - 604800
        elif t_k < -302400:
            t_k = t_k + 604800

        # Compute mean motion and eccentric anomaly with current t_k
        A = nav_dict['roota']**2
        n_0 = np.sqrt(mu / A**3)
        n = n_0 + nav_dict['deltan']
        M_k = nav_dict['M0'] + n * t_k
        E_k = kepler_solve(nav_dict['ecc'], M_k, 1e-12)

        # Compute relativistic correction and satellite clock based on current E_k
        F = -4.442807633e-10
        rela = F * nav_dict['ecc'] * np.sqrt(A) * np.sin(E_k)
        sat_clk_new = nav_dict.get('af0', 0.0) + nav_dict.get('af1', 0.0) * t_k + nav_dict.get('af2', 0.0) * t_k**2 - nav_dict.get('tgd1', 0.0) + rela

        # Update transmit time estimate including satellite clock bias
        t_new = time_dict['GPST'] - 14 - (pseudorange + c * sat_clk_new) / c

        # Check convergence on satellite clock
        if abs(sat_clk_new - sat_clk) < tol_clk and abs(t_new - t) < 1e-10:
            sat_clk = sat_clk_new
            t = t_new
            break

        sat_clk = sat_clk_new
        t = t_new

    # After iteration, set t_k for final solution
    t_k = t - nav_dict['toe']
    if t_k > 302400:
        t_k = t_k - 604800
    elif t_k < -302400:
        t_k = t_k + 604800
    
    # Compute satellite position using final t_k
    A = nav_dict['roota']**2
    n_0 = np.sqrt(mu / A**3)
    n = n_0 + nav_dict['deltan']
    M_k = nav_dict['M0'] + n * t_k
    E_k = kepler_solve(nav_dict['ecc'], M_k, 1e-12)

    sin_v_k = (np.sqrt(1 - nav_dict['ecc']**2) * np.sin(E_k)) / (1 - nav_dict['ecc'] * np.cos(E_k))
    cos_v_k = (np.cos(E_k) - nav_dict['ecc']) / (1 - nav_dict['ecc'] * np.cos(E_k))
    v_k = np.arctan2(sin_v_k, cos_v_k)
    
    PHI_k = v_k + nav_dict['omega']
    delta_u_k = nav_dict['cus'] * np.sin(2*PHI_k) + nav_dict['cuc'] * np.cos(2*PHI_k)
    delta_r_k = nav_dict['crs'] * np.sin(2*PHI_k) + nav_dict['crc'] * np.cos(2*PHI_k)
    delta_i_k = nav_dict['cis'] * np.sin(2*PHI_k) + nav_dict['cic'] * np.cos(2*PHI_k)
    
    u_k = PHI_k + delta_u_k
    r_k = A * (1 - nav_dict['ecc'] * np.cos(E_k)) + delta_r_k
    i_k = nav_dict['i0'] + delta_i_k + nav_dict['idot'] * t_k
    
    x_k_aps = r_k * np.cos(u_k)
    y_k_aps = r_k * np.sin(u_k)
    
    # Satellite velocity calculations
    mk = n
    ek = mk / (1 - nav_dict['ecc'] * np.cos(E_k))
    Wk = np.sqrt(1 - nav_dict['ecc']**2) * ek / (1 - nav_dict['ecc'] * np.cos(E_k))
    Uuk = 2 * Wk * (nav_dict['cus'] * np.cos(2*PHI_k) - nav_dict['cuc'] * np.sin(2*PHI_k))
    Rrk = 2 * Wk * (nav_dict['crs'] * np.cos(2*PHI_k) - nav_dict['crc'] * np.sin(2*PHI_k))
    Iik = 2 * Wk * (nav_dict['cis'] * np.cos(2*PHI_k) - nav_dict['cic'] * np.sin(2*PHI_k))
    Uk = Wk + Uuk
    Rk = A * nav_dict['ecc'] * ek * np.sin(E_k) + Rrk
    Ik = nav_dict['idot'] + Iik
    WK = nav_dict['Omegadot'] - OMEGA_dot_e
    
    Xxk = Rk * np.cos(u_k) - r_k * Uk * np.sin(u_k)
    Yyk = Rk * np.sin(u_k) + r_k * Uk * np.cos(u_k)
    
    # Check if GEO satellite (1 <= PRN <= 5 or >= 59)
    sv_prn = obs_dict.get('svPRN', 0)
    
    if 1 <= sv_prn <= 5 or sv_prn >= 59:  # BeiDou GEO
        # Corrected calculation for GEO satellites
        OMEGA_k = nav_dict['Omega0'] + nav_dict['Omegadot'] * t_k - OMEGA_dot_e * nav_dict['toe']
        X_GK = x_k_aps * np.cos(OMEGA_k) - y_k_aps * np.cos(i_k) * np.sin(OMEGA_k)
        Y_GK = x_k_aps * np.sin(OMEGA_k) + y_k_aps * np.cos(i_k) * np.cos(OMEGA_k)
        Z_GK = y_k_aps * np.sin(i_k)
        
        # Rotation matrices for GEO satellites
        R_X = np.array([[1, 0, 0],
                [0, np.cos(np.deg2rad(-5)), np.sin(np.deg2rad(-5))],
                [0, -np.sin(np.deg2rad(-5)), np.cos(np.deg2rad(-5))]])

        # MATLAB: phi = OMEGA_dot_e * t_k + OMEGA_dot_e * pseudorange / c
        phi = OMEGA_dot_e * t_k + OMEGA_dot_e * pseudorange / c

        # Rotation about Z by phi
        R_Z = np.array([[np.cos(phi), np.sin(phi), 0],
                [-np.sin(phi), np.cos(phi), 0],
                [0, 0, 1]])

        sv_pos = R_Z @ R_X @ np.array([X_GK, Y_GK, Z_GK])

        # Derivative of rotation matrix R_Z (as in MATLAB RZ)
        RZ = np.array([[-OMEGA_dot_e * np.sin(phi), OMEGA_dot_e * np.cos(phi), 0],
                   [-OMEGA_dot_e * np.cos(phi), -OMEGA_dot_e * np.sin(phi), 0],
                   [0, 0, 1]])

        # Velocity calculations for GEO satellites (match MATLAB)
        Xs, Ys, Zs = sv_pos[0], sv_pos[1], sv_pos[2]
        wk = nav_dict['Omegadot']
        Xsv = -Ys * wk - (Yyk * np.cos(i_k) - Zs * Ik) * np.sin(OMEGA_k) + Xxk * np.cos(OMEGA_k)
        Ysv = Xs * wk + (Yyk * np.cos(i_k) - Zs * Ik) * np.cos(OMEGA_k) + Xxk * np.sin(OMEGA_k)
        Zsv = Yyk * np.sin(i_k) + y_k_aps * Ik * np.cos(i_k)

        svel = RZ @ R_X @ np.array([Xsv, Ysv, Zsv])
        Xsvel, Ysvel, Zsvel = svel[0], svel[1], svel[2]
    else:  # BeiDou MEO/IGSO
        OMEGA_k = nav_dict['Omega0'] + (nav_dict['Omegadot'] - OMEGA_dot_e) * t_k - OMEGA_dot_e * nav_dict['toe'] - OMEGA_dot_e * (pseudorange / c)
        sv_pos = np.array([
            x_k_aps * np.cos(OMEGA_k) - y_k_aps * np.cos(i_k) * np.sin(OMEGA_k),
            x_k_aps * np.sin(OMEGA_k) + y_k_aps * np.cos(i_k) * np.cos(OMEGA_k),
            y_k_aps * np.sin(i_k)
        ])
        
        Xs, Ys, Zs = sv_pos[0], sv_pos[1], sv_pos[2]
        Xsvel = -Ys*WK - (Yyk*np.cos(i_k) - Zs*Ik)*np.sin(OMEGA_k) + Xxk*np.cos(OMEGA_k)
        Ysvel = Xs*WK + (Yyk*np.cos(i_k) - Zs*Ik)*np.cos(OMEGA_k) + Xxk*np.sin(OMEGA_k)
        Zsvel = Yyk*np.sin(i_k) + y_k_aps*Ik*np.cos(i_k)
    
    # Clock error correction with relativity - match MATLAB
    F = -4.442807633e-10
    rela = F * nav_dict['ecc'] * np.sqrt(A) * np.sin(E_k)
    sat_clk = nav_dict['af0'] + nav_dict['af1']*t_k + nav_dict['af2']*t_k**2 - nav_dict.get('tgd1', 0) + rela
    # rate_clock should be af1 + af2 * t_k per MATLAB
    rate_clock = nav_dict['af1'] + nav_dict['af2'] * t_k
    delta_tsv_L1pie = nav_dict['af1'] + 2 * nav_dict['af2']
    
    return sv_pos, sat_clk, rate_clock, Xsvel, Ysvel, Zsvel, delta_tsv_L1pie


# ====================================================================================
# RINEX File Parsers (Complete Implementation)
# ====================================================================================

def load_rinex_obs(file_path):
    """
    Complete RINEX observation file loader
    Supports RINEX 3.x format for multiple GNSS constellations
    
    Parameters:
    -----------
    file_path : str
        Path to RINEX observation file
    
    Returns:
    --------
    rinex_obs : dict
        Dictionary containing observation data with structure:
        {
            'headerData': {
                'fileVer': float,
                'approPos': dict,
                'interval': float,
                'obsType': dict
            },
            'obsData': list of epoch data
        }
    """
    print(f"Loading RINEX observation file: {file_path}")
    
    # Initialize rinex structure
    rinex_obs = {
        'headerData': {
            'fileVer': None,
            'approPos': None,
            'interval': 1.0,  # default interval
            'obsType': {
                'GPS': None,
                'BEIDOU': None,
                'GLONASS': None,
                'Galileo': None,
                'QZSS': None,
                'SBAS': None,
                'typeIndexGPS': {},
                'typeIndexBEIDOU': {},
                'typeIndexGLONASS': {},
                'typeIndexGalileo': {},
                'typeIndexQZSS': {},
                'typeIndexSBAS': {}
            }
        },
        'obsData': []
    }
    
    try:
        with open(file_path, 'r') as fid:
            # Read header
            print("Reading header...")
            while True:
                line = fid.readline()
                if not line:
                    break
                
                if 'RINEX VERSION / TYPE' in line:
                    rinex_obs['headerData']['fileVer'] = str2doubleq(line[0:9])
                
                elif 'APPROX POSITION XYZ' in line:
                    approx_pos_ecef = [
                        str2doubleq(line[0:14]),
                        str2doubleq(line[14:28]),
                        str2doubleq(line[28:42])
                    ]
                    lat, lon, h = xyz_to_blh(approx_pos_ecef[0], approx_pos_ecef[1], approx_pos_ecef[2])
                    rinex_obs['headerData']['approPos'] = {
                        'approxPosECEF': approx_pos_ecef,
                        'approxPosGeo': [lat, lon, h]
                    }
                
                elif 'INTERVAL' in line:
                    rinex_obs['headerData']['interval'] = str2doubleq(line[4:10])
                
                elif 'SYS / # / OBS TYPES' in line:
                    constellation = line[0]
                    n_observables = int(str2doubleq(line[3:6]))
                    
                    # Parse observable types
                    observables = []
                    line_obs = line[7:60].split()
                    observables.extend(line_obs)
                    
                    # Read continuation lines if more than 13 observables
                    while n_observables > 13:
                        line = fid.readline()
                        line_obs = line[7:60].split()
                        observables.extend(line_obs)
                        n_observables -= 13
                    
                    # Store observables by constellation
                    if constellation == 'G':
                        rinex_obs['headerData']['obsType']['GPS'] = observables
                        for idx, obs in enumerate(observables):
                            rinex_obs['headerData']['obsType']['typeIndexGPS'][obs] = idx
                    elif constellation == 'C':
                        rinex_obs['headerData']['obsType']['BEIDOU'] = observables
                        for idx, obs in enumerate(observables):
                            rinex_obs['headerData']['obsType']['typeIndexBEIDOU'][obs] = idx
                    elif constellation == 'R':
                        rinex_obs['headerData']['obsType']['GLONASS'] = observables
                        for idx, obs in enumerate(observables):
                            rinex_obs['headerData']['obsType']['typeIndexGLONASS'][obs] = idx
                    elif constellation == 'E':
                        rinex_obs['headerData']['obsType']['Galileo'] = observables
                        for idx, obs in enumerate(observables):
                            rinex_obs['headerData']['obsType']['typeIndexGalileo'][obs] = idx
                    elif constellation == 'J':
                        rinex_obs['headerData']['obsType']['QZSS'] = observables
                        for idx, obs in enumerate(observables):
                            rinex_obs['headerData']['obsType']['typeIndexQZSS'][obs] = idx
                    elif constellation == 'S':
                        rinex_obs['headerData']['obsType']['SBAS'] = observables
                        for idx, obs in enumerate(observables):
                            rinex_obs['headerData']['obsType']['typeIndexSBAS'][obs] = idx
                
                elif 'END OF HEADER' in line:
                    break
            
            # Read observation data
            print("Reading observations...")
            obs_count = 0
            
            while True:
                line = fid.readline()
                if not line:
                    break
                
                # Check for epoch line
                if len(line) > 0 and line[0] == '>':
                    # Parse epoch time
                    year = int(str2doubleq(line[2:6]))
                    month = int(str2doubleq(line[7:9]))
                    day = int(str2doubleq(line[10:12]))
                    hour = int(str2doubleq(line[13:15]))
                    minute = int(str2doubleq(line[16:18]))
                    second = str2doubleq(line[18:29])
                    
                    gpst, gps_week = date_to_gps_time(year, month, day, hour + minute/60 + second/3600)
                    
                    epoch_flag = int(str2doubleq(line[31]))
                    n_sat = int(str2doubleq(line[32:35]))
                    
                    epoch_data = {
                        'time': {
                            'GPSWeek': gps_week,
                            'GPST': gpst,
                            'year': year,
                            'month': month,
                            'day': day,
                            'hour': hour,
                            'min': minute,
                            'sec': second
                        },
                        'obsInfo': {
                            'epochFlag': epoch_flag,
                            'nSat': n_sat
                        },
                        'svObs': [],
                        'svPRNIndex': {
                            'GPS': {},
                            'BEIDOU': {},
                            'GLONASS': {},
                            'Galileo': {},
                            'QZSS': {},
                            'SBAS': {}
                        }
                    }
                    
                    # Read satellite observations
                    for sv_idx in range(n_sat):
                        line = fid.readline()
                        if not line or len(line) < 3:
                            continue
                        
                        constellation_code = line[0]
                        sv_prn = int(str2doubleq(line[1:3]))
                        
                        # Map constellation code to name
                        const_map = {
                            'G': 'GPS',
                            'C': 'BEIDOU',
                            'R': 'GLONASS',
                            'E': 'Galileo',
                            'J': 'QZSS',
                            'S': 'SBAS'
                        }
                        
                        constellation = const_map.get(constellation_code, 'UNKNOWN')
                        if constellation == 'UNKNOWN':
                            continue
                        
                        # Get observable types for this constellation
                        obs_types = rinex_obs['headerData']['obsType'].get(constellation)
                        if obs_types is None:
                            continue
                        
                        n_measurements = len(obs_types)
                        
                        # Parse measurements
                        measurements = {}
                        for meas_idx in range(n_measurements):
                            start_col = 3 + meas_idx * 16
                            end_col = start_col + 14
                            
                            if end_col <= len(line):
                                meas_str = line[start_col:end_col].strip()
                                if meas_str:
                                    try:
                                        meas_val = str2doubleq(meas_str)
                                        if meas_val != 0 and not np.isnan(meas_val):
                                            measurements[obs_types[meas_idx]] = meas_val
                                    except:
                                        pass
                        
                        sv_obs = {
                            'constellation': constellation,
                            'svPRN': sv_prn,
                            'measurements': measurements
                        }
                        
                        epoch_data['svObs'].append(sv_obs)
                        epoch_data['svPRNIndex'][constellation][sv_prn] = sv_idx
                    
                    rinex_obs['obsData'].append(epoch_data)
                    obs_count += 1
                    
                    if obs_count % 100 == 0:
                        print(f"  Read {obs_count} epochs, TOW: {gpst:.3f}")
            
            print(f"Completed! Read {obs_count} epochs")
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading RINEX observation file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return rinex_obs


def load_rinex_nav(file_path):
    """
    Complete RINEX navigation file loader
    Supports RINEX 3.x format for GPS, BeiDou, and QZSS
    
    Parameters:
    -----------
    file_path : str
        Path to RINEX navigation file
    
    Returns:
    --------
    rinex_nav : dict
        Dictionary containing navigation data with structure:
        {
            'headerData': {
                'fileVer': float,
                'ionosphericCorr': dict
            },
            'navData': {
                'GPS': dict,
                'BEIDOU': dict,
                'QZSS': dict
            }
        }
    """
    print(f"Loading RINEX navigation file: {file_path}")
    
    # Initialize rinex structure
    rinex_nav = {
        'headerData': {
            'fileVer': None,
            'ionosphericCorr': {
                'GPSA': None,
                'GPSB': None,
                'BDSA': None,
                'BDSB': None,
                'GAL': None,
                'QZSA': None,
                'QZSB': None,
                'IRNA': None,
                'IRNB': None
            }
        },
        'navData': {
            'GPS': {},
            'BEIDOU': {},
            'GLONASS': {},
            'Galileo': {},
            'QZSS': {},
            'SBAS': {},
            'IRNSS': {}
        }
    }
    
    try:
        with open(file_path, 'r') as fid:
            # Read header
            print("Reading header...")
            while True:
                line = fid.readline()
                if not line:
                    break
                
                if 'RINEX VERSION / TYPE' in line:
                    rinex_nav['headerData']['fileVer'] = str2doubleq(line[0:9])
                
                elif 'IONOSPHERIC CORR' in line:
                    iono_corr_type = line[0:4].strip()
                    if iono_corr_type == 'GAL':
                        iono_corr_type = 'GAL'
                    
                    iono_msg = []
                    for i in range(4):
                        iono_msg.append(str2doubleq(line[5 + i*12:17 + i*12]))
                    
                    rinex_nav['headerData']['ionosphericCorr'][iono_corr_type] = iono_msg
                
                elif 'END OF HEADER' in line:
                    break
            
            # Read navigation data
            print("Reading navigation messages...")
            nav_count = 0
            
            while True:
                line = fid.readline()
                if not line:
                    break
                
                if len(line) < 4:
                    continue
                
                constellation = line[0]
                
                try:
                    sv_prn = int(str2doubleq(line[1:3]))
                except:
                    continue
                
                # Parse based on constellation
                if constellation == 'G':  # GPS
                    sv_prn_str = f"GPS{sv_prn:03d}"
                    navmsg = parse_gps_nav_message(line, fid)
                    
                    if sv_prn_str not in rinex_nav['navData']['GPS']:
                        rinex_nav['navData']['GPS'][sv_prn_str] = []
                    rinex_nav['navData']['GPS'][sv_prn_str].append(navmsg)
                    nav_count += 1
                
                elif constellation == 'C':  # BeiDou
                    sv_prn_str = f"BEIDOU{sv_prn:03d}"
                    navmsg = parse_beidou_nav_message(line, fid)
                    
                    if sv_prn_str not in rinex_nav['navData']['BEIDOU']:
                        rinex_nav['navData']['BEIDOU'][sv_prn_str] = []
                    rinex_nav['navData']['BEIDOU'][sv_prn_str].append(navmsg)
                    nav_count += 1
                
                elif constellation == 'J':  # QZSS
                    sv_prn_qzss = sv_prn + 32
                    sv_prn_str = f"QZSS{sv_prn_qzss:03d}"
                    navmsg = parse_gps_nav_message(line, fid)  # QZSS uses GPS format
                    
                    if sv_prn_str not in rinex_nav['navData']['QZSS']:
                        rinex_nav['navData']['QZSS'][sv_prn_str] = []
                    rinex_nav['navData']['QZSS'][sv_prn_str].append(navmsg)
                    nav_count += 1
                
                elif constellation == 'R':  # GLONASS - skip
                    for _ in range(3):
                        fid.readline()
                
                elif constellation == 'E':  # Galileo - skip
                    for _ in range(7):
                        fid.readline()
                
                elif constellation == 'S':  # SBAS - skip
                    for _ in range(3):
                        fid.readline()
                
                elif constellation == 'I':  # IRNSS - skip
                    for _ in range(7):
                        fid.readline()
            
            print(f"Completed! Read {nav_count} navigation messages")
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading RINEX navigation file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return rinex_nav


def parse_gps_nav_message(first_line, fid):
    """
    Parse GPS/QZSS navigation message from RINEX file
    
    Parameters:
    -----------
    first_line : str
        First line of navigation message
    fid : file object
        File handle
    
    Returns:
    --------
    navmsg : dict
        Navigation message dictionary
    """
    navmsg = {}
    
    # Line 1: PRN, epoch, clock
    navmsg['year'] = int(str2doubleq(first_line[3:8]))
    navmsg['month'] = int(str2doubleq(first_line[8:11]))
    navmsg['day'] = int(str2doubleq(first_line[11:14]))
    navmsg['hour'] = int(str2doubleq(first_line[14:17]))
    navmsg['min'] = int(str2doubleq(first_line[17:20]))
    navmsg['sec'] = str2doubleq(first_line[20:23])
    
    toc, gps_week = date_to_gps_time(navmsg['year'], navmsg['month'], navmsg['day'],
                                      navmsg['hour'] + navmsg['min']/60 + navmsg['sec']/3600)
    navmsg['toc'] = toc
    navmsg['GPSWeek'] = gps_week
    
    navmsg['af0'] = str2doubleq(first_line[23:42])
    navmsg['af1'] = str2doubleq(first_line[42:61])
    navmsg['af2'] = str2doubleq(first_line[61:80])
    
    # Line 2: Broadcast orbit 1
    line = fid.readline()
    navmsg['IODE'] = str2doubleq(line[4:23])
    navmsg['crs'] = str2doubleq(line[23:42])
    navmsg['deltan'] = str2doubleq(line[42:61])
    navmsg['M0'] = str2doubleq(line[61:80])
    
    # Line 3: Broadcast orbit 2
    line = fid.readline()
    navmsg['cuc'] = str2doubleq(line[4:23])
    navmsg['ecc'] = str2doubleq(line[23:42])
    navmsg['cus'] = str2doubleq(line[42:61])
    navmsg['roota'] = str2doubleq(line[61:80])
    
    # Line 4: Broadcast orbit 3
    line = fid.readline()
    navmsg['toe'] = str2doubleq(line[4:23])
    navmsg['cic'] = str2doubleq(line[23:42])
    navmsg['Omega0'] = str2doubleq(line[42:61])
    navmsg['cis'] = str2doubleq(line[61:80])
    
    # Line 5: Broadcast orbit 4
    line = fid.readline()
    navmsg['i0'] = str2doubleq(line[4:23])
    navmsg['crc'] = str2doubleq(line[23:42])
    navmsg['omega'] = str2doubleq(line[42:61])
    navmsg['Omegadot'] = str2doubleq(line[61:80])
    
    # Line 6: Broadcast orbit 5
    line = fid.readline()
    navmsg['idot'] = str2doubleq(line[4:23])
    navmsg['codesOnL2'] = str2doubleq(line[23:42])
    navmsg['weekno'] = str2doubleq(line[42:61])
    navmsg['L2flag'] = str2doubleq(line[61:80])
    
    # Line 7: SV accuracy, health, TGD, IODC
    line = fid.readline()
    navmsg['svaccur'] = str2doubleq(line[4:23])
    navmsg['svhealth'] = str2doubleq(line[23:42])
    navmsg['tgd1'] = str2doubleq(line[42:61])
    navmsg['IODC'] = str2doubleq(line[61:80])
    
    # Line 8: Transmission time, fit interval
    line = fid.readline()
    navmsg['tom'] = str2doubleq(line[4:23])
    navmsg['fit'] = str2doubleq(line[23:42])
    
    return navmsg


def parse_beidou_nav_message(first_line, fid):
    """
    Parse BeiDou navigation message from RINEX file
    
    Parameters:
    -----------
    first_line : str
        First line of navigation message
    fid : file object
        File handle
    
    Returns:
    --------
    navmsg : dict
        Navigation message dictionary
    """
    navmsg = {}
    
    # Line 1: PRN, epoch, clock
    navmsg['year'] = int(str2doubleq(first_line[3:8]))
    navmsg['month'] = int(str2doubleq(first_line[8:11]))
    navmsg['day'] = int(str2doubleq(first_line[11:14]))
    navmsg['hour'] = int(str2doubleq(first_line[14:17]))
    navmsg['min'] = int(str2doubleq(first_line[17:20]))
    navmsg['sec'] = str2doubleq(first_line[20:23])
    
    toc, gps_week = date_to_gps_time(navmsg['year'], navmsg['month'], navmsg['day'],
                                      navmsg['hour'] + navmsg['min']/60 + navmsg['sec']/3600)
    navmsg['toc'] = toc
    navmsg['GPSWeek'] = gps_week
    
    navmsg['af0'] = str2doubleq(first_line[23:42])
    navmsg['af1'] = str2doubleq(first_line[42:61])
    navmsg['af2'] = str2doubleq(first_line[61:80])
    
    # Line 2: Broadcast orbit 1
    line = fid.readline()
    navmsg['IODE'] = str2doubleq(line[4:23])
    navmsg['crs'] = str2doubleq(line[23:42])
    navmsg['deltan'] = str2doubleq(line[42:61])
    navmsg['M0'] = str2doubleq(line[61:80])
    
    # Line 3: Broadcast orbit 2
    line = fid.readline()
    navmsg['cuc'] = str2doubleq(line[4:23])
    navmsg['ecc'] = str2doubleq(line[23:42])
    navmsg['cus'] = str2doubleq(line[42:61])
    navmsg['roota'] = str2doubleq(line[61:80])
    
    # Line 4: Broadcast orbit 3
    line = fid.readline()
    navmsg['toe'] = str2doubleq(line[4:23])
    navmsg['cic'] = str2doubleq(line[23:42])
    navmsg['Omega0'] = str2doubleq(line[42:61])
    navmsg['cis'] = str2doubleq(line[61:80])
    
    # Line 5: Broadcast orbit 4
    line = fid.readline()
    navmsg['i0'] = str2doubleq(line[4:23])
    navmsg['crc'] = str2doubleq(line[23:42])
    navmsg['omega'] = str2doubleq(line[42:61])
    navmsg['Omegadot'] = str2doubleq(line[61:80])
    
    # Line 6: Broadcast orbit 5
    line = fid.readline()
    navmsg['idot'] = str2doubleq(line[4:23])
    # Reserved field for BeiDou
    navmsg['weekno'] = str2doubleq(line[42:61])
    
    # Line 7: SV accuracy, health, TGD1, TGD2
    line = fid.readline()
    navmsg['svaccur'] = str2doubleq(line[4:23])
    navmsg['svhealth'] = str2doubleq(line[23:42])
    navmsg['tgd1'] = str2doubleq(line[42:61])
    navmsg['tgd2'] = str2doubleq(line[61:80])
    
    # Line 8: Transmission time, IODC
    line = fid.readline()
    navmsg['tom'] = str2doubleq(line[4:23])
    navmsg['IODC'] = str2doubleq(line[23:42])
    
    return navmsg


# ====================================================================================
# Main BDS Data Preprocessing Function
# ====================================================================================

def bds_data_preprocess(file_path_dict, approx_coor, alpha, beta, doy):
    """
    Main function to preprocess BeiDou observation data
    
    Parameters:
    -----------
    file_path_dict : dict
        Dictionary with keys 'obs' and 'nav' containing file paths
    approx_coor : array-like
        Approximate receiver coordinates [X, Y, Z] in ECEF (meters)
    alpha : array-like
        4 alpha coefficients for ionospheric correction
    beta : array-like
        4 beta coefficients for ionospheric correction
    doy : int
        Day of year
    
    Returns:
    --------
    bds_data : ndarray
        Processed BeiDou data (N x 15 array)
        Columns: [epoch, GPS_week, GPS_TOW, sat_PRN, pseudorange_corrected,
                  sat_X, sat_Y, sat_Z, pseudorange_rate,
                  sat_vel_X, sat_vel_Y, sat_vel_Z, SNR, elevation, azimuth]
    """
    # Constants
    c = 299792458.0  # Speed of light
    B1I = 1561098000.0  # BeiDou B1I frequency
    B2a = 1176450000.0  # BeiDou B2a frequency
    B2I = 1207140000.0  # BeiDou B2I frequency
    
    bds_data = []
    
    # Load observation and navigation files
    print("Loading RINEX files...")
    obs_data = load_rinex_obs(file_path_dict['obs'])
    nav_data = load_rinex_nav(file_path_dict['nav'])
    
    # Calculate rotation matrix (ECEF to local geodetic frame)
    lat, lon, height = xyz_to_blh(approx_coor[0], approx_coor[1], approx_coor[2])
    rota = np.array([
        [-sind(lat)*cosd(lon), -sind(lon), cosd(lat)*cosd(lon)],
        [-sind(lat)*sind(lon), cosd(lon), cosd(lat)*sind(lon)],
        [cosd(lat), 0, sind(lat)]
    ])
    
    # Calculate zenith tropospheric delay
    ZHD, ZWD = zenith_delay(lat, height, doy)
    
    print(f"Processing observations...")
    print(f"Receiver position: Lat={lat:.6f}°, Lon={lon:.6f}°, H={height:.2f}m")
    
    # Process each epoch
    epoch_num = len(obs_data['obsData'])
    
    for i in range(epoch_num):
        epoch = obs_data['obsData'][i]
        
        # Skip if epoch flag is not 0 or less than 4 satellites
        if epoch['obsInfo']['epochFlag'] != 0 or epoch['obsInfo']['nSat'] < 4:
            continue
        
        # Data storage for this epoch
        data_ = np.zeros((epoch['obsInfo']['nSat'], 15))  # Match MATLAB structure
        
        # Process each satellite
        valid_sat_count = 0
        for j in range(epoch['obsInfo']['nSat']):
            sv_obs = epoch['svObs'][j]
            
            # Only process BeiDou satellites
            if sv_obs['constellation'] != 'BEIDOU':
                continue
            
            # Exclude problematic satellites
            sv_prn = sv_obs['svPRN']
            if sv_prn in [56, 58]:
                continue
            
            # Get navigation data for this satellite
            sv_prn_str = f"BEIDOU{sv_prn:03d}"
            if sv_prn_str not in nav_data['navData']['BEIDOU']:
                print(f"Warning: Satellite {sv_prn_str} not in navigation file, skipping")
                continue

            nav_all = nav_data['navData']['BEIDOU'][sv_prn_str]

            # Determine pseudorange measurement for more accurate ephemeris selection
            pseudo_meas = None
            if 'C2I' in sv_obs['measurements'] and not np.isnan(sv_obs['measurements'].get('C2I', np.nan)):
                pseudo_meas = sv_obs['measurements'].get('C2I')
            elif 'C1I' in sv_obs['measurements'] and not np.isnan(sv_obs['measurements'].get('C1I', np.nan)):
                pseudo_meas = sv_obs['measurements'].get('C1I')

            # Select ephemeris closest in transmission time (uses pseudorange when available)
            if isinstance(nav_all, list) and len(nav_all) > 1:
                # If pseudorange available, compute transmission time t and choose nav minimizing |t-nav.toe|
                if pseudo_meas is not None:
                    c = 299792458.0
                    t_tx = epoch['time']['GPST'] - 14 - pseudo_meas / c
                    tks = [abs(t_tx - nav['toe']) for nav in nav_all]
                    idx = int(np.argmin(np.array(tks)))
                    nav = nav_all[idx]
                else:
                    toe_list = [nav['toe'] for nav in nav_all]
                    idx = int(np.argmin(np.abs(np.array(toe_list) - epoch['time']['GPST'])))
                    nav = nav_all[idx]
            else:
                nav = nav_all if not isinstance(nav_all, list) else nav_all[0]
            
            # Calculate satellite position
            obs_dict = {
                'svPRN': sv_prn,
                'C2I': sv_obs['measurements'].get('C2I', np.nan) if 'C2I' in sv_obs['measurements'] else np.nan,
                'C1I': sv_obs['measurements'].get('C1I', np.nan) if 'C1I' in sv_obs['measurements'] else np.nan,
                'D2I': sv_obs['measurements'].get('D2I', np.nan) if 'D2I' in sv_obs['measurements'] else np.nan,
                'D1I': sv_obs['measurements'].get('D1I', np.nan) if 'D1I' in sv_obs['measurements'] else np.nan,
                'S2I': sv_obs['measurements'].get('S2I', np.nan) if 'S2I' in sv_obs['measurements'] else np.nan,
                'S1I': sv_obs['measurements'].get('S1I', np.nan) if 'S1I' in sv_obs['measurements'] else np.nan,
            }
            
            result = cal_sv_pos_beidou(epoch['time'], obs_dict, nav, obs_data['headerData'])
            
            if result[0] is None:
                continue
            
            sv_pos, sat_clk, rate_clock, Xsvel, Ysvel, Zsvel, delta_tsv_L1pie = result
            
            # Calculate elevation and azimuth - match MATLAB implementation
            sat_xyz = (sv_pos - approx_coor) @ rota
            ele = asind(sat_xyz[2] / np.linalg.norm(sat_xyz))
            # MATLAB uses atan2d(Sat_xyz(1),Sat_xyz(2)) so we call
            # atan2d(y,x) with y=Sat_xyz(0)=x and x=Sat_xyz(1)=y to replicate
            azimut = atan2d(sat_xyz[0], sat_xyz[1])
            
            # Calculate ionospheric delay
            T_iono, D_iono, Std_iono, Var_iono = ionospheric_delay(
                epoch['time']['GPST'], lat, lon, azimut, ele, alpha, beta
            )
            
            # Calculate tropospheric delay (Neill mapping expects latitude in degrees)
            trop = neill_map(ZHD, ZWD, ele, lat, height, doy)
            
            # Get range and rate measurements - Fixed to match MATLAB logic completely
            range_val = np.nan
            rate = 0.0
            snr = 0.0
            
            # Match MATLAB logic for selecting measurements exactly
            obsHeader = obs_data['headerData']
            if ('C2I' in obsHeader['obsType']['typeIndexBEIDOU'] and 
                'C2I' in sv_obs['measurements'] and 
                not np.isnan(sv_obs['measurements']['C2I'])):
                range_val = sv_obs['measurements']['C2I']
                if 'D2I' in sv_obs['measurements'] and not np.isnan(sv_obs['measurements']['D2I']):
                    rate = -1.0 * sv_obs['measurements']['D2I'] * c / B1I
                if 'S2I' in sv_obs['measurements']:
                    snr = sv_obs['measurements']['S2I']
            elif ('C1I' in obsHeader['obsType']['typeIndexBEIDOU'] and 
                  'C1I' in sv_obs['measurements'] and 
                  not np.isnan(sv_obs['measurements']['C1I'])):
                range_val = sv_obs['measurements']['C1I']
                if 'D1I' in sv_obs['measurements'] and not np.isnan(sv_obs['measurements']['D1I']):
                    rate = -1.0 * sv_obs['measurements']['D1I'] * c / B1I
                if 'S1I' in sv_obs['measurements']:
                    snr = sv_obs['measurements']['S1I']
            else:
                continue
            
            # Corrected pseudorange - Fixed to match MATLAB exactly
            L_noclk = range_val - trop - D_iono + c * sat_clk
            
            # Store data: [epoch, GPS_week, GPS_TOW, sat_PRN, pseudorange_corrected,
            #              sat_X, sat_Y, sat_Z, pseudorange_rate,
            #              sat_vel_X, sat_vel_Y, sat_vel_Z, SNR, elevation, azimuth]
            data_[valid_sat_count, :] = [
                i + 1,
                epoch['time']['GPSWeek'],
                epoch['time']['GPST'],
                sv_prn,
                L_noclk,
                sv_pos[0], sv_pos[1], sv_pos[2],
                rate,
                Xsvel, Ysvel, Zsvel,
                snr,
                ele,
                azimut
            ]
            
            valid_sat_count += 1
        
        # Only add data if we have valid satellites
        if valid_sat_count > 0:
            bds_data.extend(data_[:valid_sat_count])
    
    # Convert to numpy array
    if bds_data:
        bds_data = np.array(bds_data)
    else:
        bds_data = np.array([]).reshape(0, 15)
    
    # Save to Excel file
    obs_folder = os.path.dirname(file_path_dict['obs'])
    output_file = os.path.join(obs_folder, 'BDSData_output.xlsx')
    
    print(f"\nSaving results to {output_file}...")
    if len(bds_data) > 0:
        df = pd.DataFrame(bds_data, columns=[
            'Epoch', 'GPS_Week', 'GPS_TOW', 'Sat_PRN', 'Pseudorange_Corrected',
            'Sat_X', 'Sat_Y', 'Sat_Z', 'Pseudorange_Rate',
            'Sat_Vel_X', 'Sat_Vel_Y', 'Sat_Vel_Z', 'SNR', 'Elevation', 'Azimuth'
        ])
        # 不写入表头（只输出数据行）
        df.to_excel(output_file, index=False, header=False)
    else:
        # Create empty file if no data
        pd.DataFrame().to_excel(output_file, index=False, header=False)
    
    print(f"Processing complete! {len(bds_data)} observations processed.")
    
    return bds_data


# ====================================================================================
# Example Usage
# ====================================================================================

if __name__ == "__main__":
    """
    Example usage of the BDS data preprocessing function
    """
    
    # Interactive usage: allow file selection and console inputs
    try:
        import tkinter as tk
        from tkinter import filedialog
        tk_root = tk.Tk()
        tk_root.withdraw()
        use_file_dialog = True
    except Exception:
        use_file_dialog = False

    print("BDS Data Preprocessing - Interactive Mode")
    print("=" * 60)

    if use_file_dialog:
        print("请选择观测文件（RINEX observation）...")
        obs_path = filedialog.askopenfilename(title="Select RINEX observation file", filetypes=[("RINEX obs", "*.24O *.24o *.obs *.rnx *.rn3"), ("All files", "*.*")])
        print("请选择星历文件（RINEX navigation）...")
        nav_path = filedialog.askopenfilename(title="Select RINEX navigation file", filetypes=[("RINEX nav", "*.24P *.24p *.nav *.rnx"), ("All files", "*.*")])
    else:
        obs_path = input("输入观测文件路径 (obs): ").strip()
        nav_path = input("输入星历文件路径 (nav): ").strip()

    if not obs_path or not nav_path:
        print("未提供文件路径，退出。")
    else:
        # Approximate receiver position
        default_pos = "-2191255.98 5198712.24 2965449.53"
        pos_str = input(f"输入接收机近似ECEF坐标 X Y Z（以空格分隔），回车使用默认 [{default_pos}]: ").strip()
        import re
        if not pos_str:
            pos_vals = [float(x) for x in re.split(r'[,\s]+', default_pos.strip()) if x]
        else:
            try:
                parts = [x for x in re.split(r'[,\s]+', pos_str.strip()) if x]
                pos_vals = [float(x) for x in parts]
                if len(pos_vals) != 3:
                    raise ValueError
            except Exception:
                print("坐标格式错误，使用默认值。")
                pos_vals = [float(x) for x in re.split(r'[,\s]+', default_pos.strip()) if x]

        # Ionospheric parameters alpha (4 values)
        default_alpha = "1.955777406692505e-08 -1.490116119384766e-08 -1.192092895507813e-07 1.788139343261719e-07"
        a_str = input(f"输入4个电离层 alpha 参数（空格分隔），回车使用默认 [{default_alpha}]: ").strip()
        if not a_str:
            alpha = [float(x) for x in re.split(r'[,\s]+', default_alpha.strip()) if x]
        else:
            try:
                parts = [x for x in re.split(r'[,\s]+', a_str.strip()) if x]
                alpha = [float(x) for x in parts]
                if len(alpha) != 4:
                    raise ValueError
            except Exception:
                print("alpha 参数格式错误，使用默认值。")
                alpha = [float(x) for x in re.split(r'[,\s]+', default_alpha.strip()) if x]

        # Ionospheric parameters beta (4 values)
        default_beta = "1.290240000000000e+05 -1.474560000000000e+05 0.000000000000000e+00 -6.553600000000000e+04"
        b_str = input(f"输入4个电离层 beta 参数（空格分隔），回车使用默认 [{default_beta}]: ").strip()
        if not b_str:
            beta = [float(x) for x in re.split(r'[,\s]+', default_beta.strip()) if x]
        else:
            try:
                parts = [x for x in re.split(r'[,\s]+', b_str.strip()) if x]
                beta = [float(x) for x in parts]
                if len(beta) != 4:
                    raise ValueError
            except Exception:
                print("beta 参数格式错误，使用默认值。")
                beta = [float(x) for x in re.split(r'[,\s]+', default_beta.strip()) if x]

        # Day of year (DOY)
        doy_str = input("输入年积日（DOY），回车使用默认 [1]: ").strip()
        try:
            doy = int(doy_str) if doy_str else 1
        except Exception:
            print("DOY 格式错误，使用默认 1。")
            doy = 328

        file_paths = {'obs': obs_path, 'nav': nav_path}

        print('\n开始处理...')
        try:
            bds_data = bds_data_preprocess(file_paths, pos_vals, alpha, beta, doy)
            print('处理完成。')
        except Exception as e:
            print(f'处理出错: {e}')

    print("\n脚本结束。可重新运行进行新的处理。")
