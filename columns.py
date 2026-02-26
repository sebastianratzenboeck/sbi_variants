"""
Column definitions for the mock galaxy SimFormer training pipeline.

Shared between prepare_data.py and train_mock_galaxy.py to ensure
consistent column ordering across data preparation and training.
"""

# ---------------------------------------------------------------------------
# Intrinsic stellar parameters (no errors, always observed)
# ---------------------------------------------------------------------------
# Sky position is stored as 3D Cartesian unit vector (ux, uy, uz)
# converted from Galactic coordinates (glon, glat) to avoid
# periodicity at l=0/360 and pole singularities at b=±90.
# See prepare_data.galactic_to_unitvec() for the conversion.
INTRINSIC_COLS = [
    'sky_ux', 'sky_uy', 'sky_uz', 'feh', 'm_init', 'logAge', 'rad', 'logL', 'logT', 'logg', 'Av',
]

# ---------------------------------------------------------------------------
# True (noise-free) magnitudes from simulation
# ---------------------------------------------------------------------------
TRUE_MAG_COLS = [
    'GAIA_GAIA3.Gbp_mag', 'GAIA_GAIA3.G_mag', 'GAIA_GAIA3.Grp_mag',
    '2MASS_H_mag', '2MASS_J_mag', '2MASS_Ks_mag',
    'WISE_WISE.W1_mag', 'WISE_WISE.W2_mag',
    'PS1_g_mag', 'PS1_i_mag', 'PS1_r_mag', 'PS1_y_mag', 'PS1_z_mag',
    'CTIO_DECam.g_mag', 'CTIO_DECam.r_mag', 'CTIO_DECam.i_mag',
    'CTIO_DECam.z_mag', 'CTIO_DECam.Y_mag',
]

# ---------------------------------------------------------------------------
# Observed measurements (may be NaN if unobserved)
# ---------------------------------------------------------------------------
OBS_COLS = [
    # Gaia obs (already present in raw data)
    'GAIA_GAIA3.Gbp_mag_obs', 'GAIA_GAIA3.G_mag_obs', 'GAIA_GAIA3.Grp_mag_obs',
    'parallax_obs', # 'distance_obs',
    # Other surveys
    '2MASS_H_mag_obs', '2MASS_J_mag_obs', '2MASS_Ks_mag_obs',
    'WISE_WISE.W1_mag_obs', 'WISE_WISE.W2_mag_obs',
    'PS1_g_mag_obs', 'PS1_i_mag_obs', 'PS1_r_mag_obs', 'PS1_y_mag_obs', 'PS1_z_mag_obs',
    'CTIO_DECam.g_mag_obs', 'CTIO_DECam.r_mag_obs', 'CTIO_DECam.i_mag_obs',
    'CTIO_DECam.z_mag_obs', 'CTIO_DECam.Y_mag_obs',
]

# ---------------------------------------------------------------------------
# Error columns corresponding to OBS_COLS (same order)
# ---------------------------------------------------------------------------
OBS_ERR_COLS = [
    'GAIA_GAIA3.Gbp_mag_err', 'GAIA_GAIA3.G_mag_err', 'GAIA_GAIA3.Grp_mag_err',
    'parallax_err', # 'parallax_err',  # distance_obs uses parallax_err as proxy
    '2MASS_H_mag_err', '2MASS_J_mag_err', '2MASS_Ks_mag_err',
    'WISE_WISE.W1_mag_err', 'WISE_WISE.W2_mag_err',
    'PS1_g_mag_err', 'PS1_i_mag_err', 'PS1_r_mag_err', 'PS1_y_mag_err', 'PS1_z_mag_err',
    'CTIO_DECam.g_mag_err', 'CTIO_DECam.r_mag_err', 'CTIO_DECam.i_mag_err',
    'CTIO_DECam.z_mag_err', 'CTIO_DECam.Y_mag_err',
]

# ---------------------------------------------------------------------------
# Survey definitions for synthetic error generation
# {survey_name: (true_magnitude_columns, fixed_error_sigma)}
# ---------------------------------------------------------------------------
SURVEY_ERRORS = {
    '2MASS': (['2MASS_H_mag', '2MASS_J_mag', '2MASS_Ks_mag'], 0.03),
    'WISE':  (['WISE_WISE.W1_mag', 'WISE_WISE.W2_mag'], 0.05),
    'PS1':   (['PS1_g_mag', 'PS1_i_mag', 'PS1_r_mag', 'PS1_y_mag', 'PS1_z_mag'], 0.02),
    'DECam': (['CTIO_DECam.g_mag', 'CTIO_DECam.r_mag', 'CTIO_DECam.i_mag',
               'CTIO_DECam.z_mag', 'CTIO_DECam.Y_mag'], 0.02),
}

# ---------------------------------------------------------------------------
# Here more for save keeping rather then actual use
# ---------------------------------------------------------------------------
SIGMOID_SETTINGS = {
    "decaps": {"lo_pct": 0.1, "hi_pct": 99.8, "w_bright": 0.15, "w_faint": 0.05},
    "ps1":    {"lo_pct": 0.1, "hi_pct": 99.8, "w_bright": 0.15, "w_faint": 0.05},
    "2mass":  {"lo_pct": 0.05, "hi_pct": 99.5, "w_bright": 1.0,  "w_faint": 0.05},
    "unwise": {"lo_pct": 0.2,  "hi_pct": 99.5, "w_bright": 0.1,  "w_faint": 0.05},
}

# ---------------------------------------------------------------------------
# Color feature definitions
# Each tuple: (color_name, mag1_col, mag2_col) where color = mag1 - mag2
# Colors are computed from observed magnitudes
# ---------------------------------------------------------------------------
COLOR_DEFINITIONS = [
    # Gaia colors
    ("color_BP_RP", "GAIA_GAIA3.Gbp_mag_obs", "GAIA_GAIA3.Grp_mag_obs"),
    ("color_BP_G", "GAIA_GAIA3.Gbp_mag_obs", "GAIA_GAIA3.G_mag_obs"),
    ("color_G_RP", "GAIA_GAIA3.G_mag_obs", "GAIA_GAIA3.Grp_mag_obs"),

    # 2MASS colors
    ("color_J_H", "2MASS_J_mag_obs", "2MASS_H_mag_obs"),
    ("color_H_Ks", "2MASS_H_mag_obs", "2MASS_Ks_mag_obs"),
    ("color_J_Ks", "2MASS_J_mag_obs", "2MASS_Ks_mag_obs"),

    # WISE colors
    ("color_W1_W2", "WISE_WISE.W1_mag_obs", "WISE_WISE.W2_mag_obs"),
    ("color_Ks_W1", "2MASS_Ks_mag_obs", "WISE_WISE.W1_mag_obs"),
    ("color_Ks_W2", "2MASS_Ks_mag_obs", "WISE_WISE.W2_mag_obs"),

    # PS1 colors (adjacent)
    ("color_g_r_ps1", "PS1_g_mag_obs", "PS1_r_mag_obs"),
    ("color_r_i_ps1", "PS1_r_mag_obs", "PS1_i_mag_obs"),
    ("color_i_z_ps1", "PS1_i_mag_obs", "PS1_z_mag_obs"),
    ("color_z_y_ps1", "PS1_z_mag_obs", "PS1_y_mag_obs"),
    # PS1 wide-baseline colors (robust to missing intermediate bands, lower noise)
    ("color_g_i_ps1", "PS1_g_mag_obs", "PS1_i_mag_obs"),
    ("color_g_z_ps1", "PS1_g_mag_obs", "PS1_z_mag_obs"),
    ("color_g_y_ps1", "PS1_g_mag_obs", "PS1_y_mag_obs"),
    ("color_r_z_ps1", "PS1_r_mag_obs", "PS1_z_mag_obs"),
    ("color_r_y_ps1", "PS1_r_mag_obs", "PS1_y_mag_obs"),

    # DECam colors (adjacent)
    ("color_g_r_decam", "CTIO_DECam.g_mag_obs", "CTIO_DECam.r_mag_obs"),
    ("color_r_i_decam", "CTIO_DECam.r_mag_obs", "CTIO_DECam.i_mag_obs"),
    ("color_i_z_decam", "CTIO_DECam.i_mag_obs", "CTIO_DECam.z_mag_obs"),
    ("color_z_Y_decam", "CTIO_DECam.z_mag_obs", "CTIO_DECam.Y_mag_obs"),
    # DECam wide-baseline colors
    ("color_g_i_decam", "CTIO_DECam.g_mag_obs", "CTIO_DECam.i_mag_obs"),
    ("color_g_z_decam", "CTIO_DECam.g_mag_obs", "CTIO_DECam.z_mag_obs"),
    ("color_g_Y_decam", "CTIO_DECam.g_mag_obs", "CTIO_DECam.Y_mag_obs"),
    ("color_r_z_decam", "CTIO_DECam.r_mag_obs", "CTIO_DECam.z_mag_obs"),
    ("color_r_Y_decam", "CTIO_DECam.r_mag_obs", "CTIO_DECam.Y_mag_obs"),

    # Cross-survey colors (Gaia-2MASS)
    ("color_G_J", "GAIA_GAIA3.G_mag_obs", "2MASS_J_mag_obs"),
    ("color_G_Ks", "GAIA_GAIA3.G_mag_obs", "2MASS_Ks_mag_obs"),
    ("color_BP_J", "GAIA_GAIA3.Gbp_mag_obs", "2MASS_J_mag_obs"),
    ("color_RP_Ks", "GAIA_GAIA3.Grp_mag_obs", "2MASS_Ks_mag_obs"),

    # Cross-survey colors (Gaia-WISE)
    ("color_G_W1", "GAIA_GAIA3.G_mag_obs", "WISE_WISE.W1_mag_obs"),
    ("color_G_W2", "GAIA_GAIA3.G_mag_obs", "WISE_WISE.W2_mag_obs"),
    ("color_RP_W1", "GAIA_GAIA3.Grp_mag_obs", "WISE_WISE.W1_mag_obs"),

    # Cross-survey colors (2MASS-WISE, good for YSOs)
    ("color_J_W1", "2MASS_J_mag_obs", "WISE_WISE.W1_mag_obs"),
    ("color_J_W2", "2MASS_J_mag_obs", "WISE_WISE.W2_mag_obs"),
    ("color_H_W2", "2MASS_H_mag_obs", "WISE_WISE.W2_mag_obs"),

    # Cross-survey colors (PS1-2MASS)
    ("color_g_J_ps1", "PS1_g_mag_obs", "2MASS_J_mag_obs"),
    ("color_r_Ks_ps1", "PS1_r_mag_obs", "2MASS_Ks_mag_obs"),

    # Cross-survey colors (DECam-2MASS)
    ("color_g_J_decam", "CTIO_DECam.g_mag_obs", "2MASS_J_mag_obs"),
    ("color_r_Ks_decam", "CTIO_DECam.r_mag_obs", "2MASS_Ks_mag_obs"),
]

# Extract just the color names for easy reference
COLOR_COLS = [c[0] for c in COLOR_DEFINITIONS]

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
ALL_VALUE_COLS = INTRINSIC_COLS + TRUE_MAG_COLS + OBS_COLS
NUM_NODES = len(ALL_VALUE_COLS)
N_INTRINSIC = len(INTRINSIC_COLS)
N_TRUE_MAG = len(TRUE_MAG_COLS)
N_COLORS = len(COLOR_COLS)
