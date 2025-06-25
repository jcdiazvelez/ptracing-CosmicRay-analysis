import numpy as np
import healpy as hp
from statsmodels import robust

def filter_chi2_array(
    chi2sum,
    method="finite",
    save_filtered=False,
    save_path="chi2sum_filtered.npz",
    range_bounds=(0.0, 100.0),
    quantile_bounds=(1, 99),
    robust_sigma=3.0,
    return_healpy_map=True
):
    """
    Applies filtering to a chi2 array and optionally returns a Healpy-compatible version.

    Parameters:
    - chi2sum (np.ndarray): Input chi² array per pixel.
    - method (str): Filtering method: 'finite', 'range', 'quantile', 'robust'.
    - save_filtered (bool): Whether to save filtered output to .npz.
    - save_path (str): Path to save filtered array and mask.
    - range_bounds (tuple): For 'range' method: (min_val, max_val).
    - quantile_bounds (tuple): For 'quantile' method: (low%, high%).
    - robust_sigma (float): Sigma threshold for 'robust' MAD method.
    - return_healpy_map (bool): Whether to return a map with hp.UNSEEN for invalid pixels.

    Returns:
    - chi2_filtered (np.ndarray): Filtered chi2 array with np.nan for invalid values.
    - valid_mask (np.ndarray): Boolean mask of valid entries.
    - chi2_healpy_map (np.ndarray): Healpy-compatible array with hp.UNSEEN in invalid pixels (if enabled).
    """
    chi2_filtered = np.full_like(chi2sum, np.nan, dtype=np.float64)
    valid_mask = np.zeros_like(chi2sum, dtype=bool)

    if method == "finite":
        valid_mask = np.isfinite(chi2sum)

    elif method == "range":
        lower, upper = range_bounds
        valid_mask = (chi2sum >= lower) & (chi2sum <= upper) & np.isfinite(chi2sum)

    elif method == "quantile":
        chi2_finite = chi2sum[np.isfinite(chi2sum)]
        q_low = np.percentile(chi2_finite, quantile_bounds[0])
        q_high = np.percentile(chi2_finite, quantile_bounds[1])
        valid_mask = (chi2sum >= q_low) & (chi2sum <= q_high) & np.isfinite(chi2sum)
        print(f"Quantile bounds: {quantile_bounds} → values in [{q_low:.3f}, {q_high:.3f}]")

    elif method == "robust":
        chi2_finite = chi2sum[np.isfinite(chi2sum)]
        med = np.median(chi2_finite)
        mad_val = robust.mad(chi2_finite)
        valid_mask = np.abs(chi2sum - med) < robust_sigma * mad_val
        print(f"Robust filter: median={med:.3f}, MAD={mad_val:.3f}, threshold={robust_sigma * mad_val:.3f}")

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: finite, range, quantile, robust.")

    chi2_filtered[valid_mask] = chi2sum[valid_mask]

    # Generate Healpy-compatible map
    chi2_healpy_map = None
    if return_healpy_map:
        chi2_healpy_map = np.full_like(chi2sum, hp.UNSEEN, dtype=np.float64)
        chi2_healpy_map[valid_mask] = chi2sum[valid_mask]

    # Save if requested
    if save_filtered:
        np.savez(save_path,
                 chi_squared=chi2_filtered,
                 valid_mask=valid_mask,
                 chi2_healpy=chi2_healpy_map)
        print(f"Filtered data saved to: {save_path}")

    return chi2_filtered, valid_mask, chi2_healpy_map


file_maps_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/maps/'
chi2sum = np.load(file_maps_dir+"chi2_realmap_pwrind-2p6_all-wei_nside32_pix1_dof10_wo-gaussian.npz")["chi_squared"]

# Apply filtering (e.g., 1st–99th percentile range)
filtered_chi2, valid_mask, chi2_healpy = filter_chi2_array(
    chi2sum,
    method="quantile",
    quantile_bounds=(1, 99),
    save_filtered=True,
    save_path=file_maps_dir+"chi2_realmap_pwrind-2p6_all-wei_nside32_pix1_dof10_wo-gaussian_filtered_quantile.npz"
)
