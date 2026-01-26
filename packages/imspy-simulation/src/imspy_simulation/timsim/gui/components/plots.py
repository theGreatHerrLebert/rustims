"""Plotting components for TimSim GUI."""

from typing import Optional, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from nicegui import ui


# Path to resource configs
# plots.py is at: imspy/simulation/timsim/gui/components/plots.py
# resources are at: imspy/simulation/resources/configs/
_RESOURCE_DIR = Path(__file__).parent.parent.parent.parent / "resources" / "configs"


def load_window_config(config_name: str) -> Optional[pd.DataFrame]:
    """Load window configuration from CSV resource file.

    Args:
        config_name: Name of the config file (e.g., 'dia_ms_ms_windows.csv')

    Returns:
        DataFrame with window configuration or None if not found
    """
    config_path = _RESOURCE_DIR / config_name
    if config_path.exists():
        try:
            return pd.read_csv(config_path)
        except Exception as e:
            print(f"Warning: Could not load {config_name}: {e}")
    return None


# Cache for scan-mobility calibration
_calibration_cache = {}

# Cache for precursor data
_precursor_cache = {}


def get_precursor_data(reference_path: str, num_frames: int = 10, max_points: int = 5000) -> Optional[dict]:
    """Load precursor data from a reference .d dataset for visualization.

    Args:
        reference_path: Path to the reference .d folder
        num_frames: Number of frames to sample precursors from
        max_points: Maximum number of points to return (for performance)

    Returns:
        Dictionary with 'mz' and 'mobility' arrays, or None if loading fails
    """
    if not reference_path:
        return None

    cache_key = (reference_path, num_frames)
    if cache_key in _precursor_cache:
        return _precursor_cache[cache_key]

    try:
        from imspy_timstof.dia import TimsDatasetDIA
        from pathlib import Path

        ref_path = Path(reference_path)
        if not ref_path.exists() or ref_path.suffix != ".d":
            return None

        # Load dataset
        dataset = TimsDatasetDIA(str(ref_path), in_memory=False, use_bruker_sdk=True)

        # Sample precursor signal
        frame = dataset.sample_precursor_signal(num_frames=num_frames, max_intensity=50.0, take_probability=0.3)

        mz = frame.mz
        mobility = frame.mobility

        # Subsample if too many points
        if len(mz) > max_points:
            indices = np.random.choice(len(mz), max_points, replace=False)
            mz = mz[indices]
            mobility = mobility[indices]

        result = {'mz': mz, 'mobility': mobility}
        _precursor_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"Warning: Could not load precursor data from {reference_path}: {e}")
        return None


def clear_precursor_cache():
    """Clear the precursor data cache."""
    _precursor_cache.clear()


def get_scan_mobility_converter(reference_path: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Load scan-to-mobility converter from a reference .d dataset.

    Args:
        reference_path: Path to the reference .d folder

    Returns:
        Function that converts scan values to 1/K0 values, or None if loading fails
    """
    if not reference_path:
        return None

    # Check cache
    if reference_path in _calibration_cache:
        return _calibration_cache[reference_path]

    try:
        from imspy_timstof.dia import TimsDatasetDIA

        ref_path = Path(reference_path)
        if not ref_path.exists() or ref_path.suffix != ".d":
            return None

        # Load dataset (not in memory for quick calibration access)
        dataset = TimsDatasetDIA(str(ref_path), in_memory=False, use_bruker_sdk=True)

        # Get scan range from first MS2 frame
        # Use frame 1 for calibration (consistent across frames for same instrument settings)
        frame_id = 1

        def scan_to_mobility(scan_values: np.ndarray) -> np.ndarray:
            """Convert scan values to 1/K0 using dataset calibration."""
            return dataset.scan_to_inverse_mobility(frame_id, scan_values.astype(np.int32))

        # Cache the converter
        _calibration_cache[reference_path] = scan_to_mobility
        return scan_to_mobility

    except Exception as e:
        print(f"Warning: Could not load calibration from {reference_path}: {e}")
        return None


def clear_calibration_cache():
    """Clear the calibration cache (call when reference path changes)."""
    _calibration_cache.clear()


def emg_pdf(x: np.ndarray, mu: float, sigma: float, k: float) -> np.ndarray:
    """Calculate EMG (Exponentially Modified Gaussian) probability density.

    Args:
        x: Array of x values
        mu: Mean of the Gaussian component
        sigma: Standard deviation of the Gaussian component
        k: Rate parameter of the exponential component (tailing)

    Returns:
        Array of PDF values
    """
    from scipy import special

    if k <= 0:
        # Pure Gaussian when k = 0
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # EMG formula
    lam = 1.0 / k
    term1 = lam / 2 * np.exp(lam / 2 * (2 * mu + lam * sigma**2 - 2 * x))
    term2 = special.erfc((mu + lam * sigma**2 - x) / (np.sqrt(2) * sigma))
    return term1 * term2


def create_emg_plot(
    sigma: float = 1.5,
    k: float = 5.0,
) -> dict:
    """Create an interactive EMG peak shape visualization.

    Args:
        sigma: Initial sigma value (peak width)
        k: Initial k value (tailing)

    Returns:
        Dictionary with plot and control elements
    """
    try:
        import matplotlib
        matplotlib.use('agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        ui.label("Matplotlib not available for plotting")
        return {}

    # Generate x values
    x = np.linspace(-5, 25, 500)
    mu = 5.0  # Fixed peak position

    # Store current values in lists for closure
    current_sigma = [sigma]
    current_k = [k]

    # Create the UI elements
    with ui.card().classes("w-full"):
        ui.label("EMG Peak Shape Preview").classes("font-bold text-lg mb-2")

        # Create plot using NiceGUI's pyplot context
        with ui.pyplot(figsize=(8, 4)) as plot_element:
            y = emg_pdf(x, mu, sigma, k)
            plt.plot(x, y, "b-", linewidth=2)
            plt.fill_between(x, y, alpha=0.3)
            plt.xlabel("Retention Time (relative)")
            plt.ylabel("Intensity (relative)")
            plt.title(f"EMG Peak Shape (σ={sigma:.2f}, k={k:.2f})")
            plt.xlim(-5, 25)
            plt.ylim(0, max(y) * 1.1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

        def update_plot():
            """Update the plot with current parameters."""
            with plot_element:
                plt.clf()
                y = emg_pdf(x, mu, current_sigma[0], current_k[0])
                plt.plot(x, y, "b-", linewidth=2)
                plt.fill_between(x, y, alpha=0.3)
                plt.xlabel("Retention Time (relative)")
                plt.ylabel("Intensity (relative)")
                plt.title(f"EMG Peak Shape (σ={current_sigma[0]:.2f}, k={current_k[0]:.2f})")
                plt.xlim(-5, 25)
                plt.ylim(0, max(y) * 1.1 if max(y) > 0 else 1)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

        with ui.row().classes("w-full gap-8 mt-4"):
            with ui.column().classes("flex-1"):
                ui.label("Sigma (peak width)")

                def on_sigma_change(e):
                    current_sigma[0] = e.value
                    update_plot()

                sigma_slider = ui.slider(
                    value=sigma,
                    min=0.1,
                    max=5.0,
                    step=0.1,
                    on_change=on_sigma_change,
                ).classes("w-full")

            with ui.column().classes("flex-1"):
                ui.label("K (tailing)")

                def on_k_change(e):
                    current_k[0] = e.value
                    update_plot()

                k_slider = ui.slider(
                    value=k,
                    min=0.0,
                    max=15.0,
                    step=0.5,
                    on_change=on_k_change,
                ).classes("w-full")

    return {
        "plot": plot_element,
        "sigma_slider": sigma_slider,
        "k_slider": k_slider,
        "update": update_plot,
    }


def create_ccs_charge_plot(
    inv_mob_std: float = 0.009,
    on_std_change: callable = None,
) -> dict:
    """Create an interactive CCS vs charge state visualization.

    Shows how CCS distributes across m/z for different charge states,
    and how the inverse mobility standard deviation affects the spread.

    Args:
        inv_mob_std: Initial inverse mobility standard deviation
        on_std_change: Optional callback when std value changes (receives new value)

    Returns:
        Dictionary with plot and control elements
    """
    try:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
    except ImportError:
        ui.label("Matplotlib not available for plotting")
        return {}

    # Store current values
    current_std = [inv_mob_std]

    # Typical m/z range for peptides
    mz_range = np.linspace(300, 1500, 100)

    # CCS approximation using square root model (simplified Mason-Schamp)
    # CCS ≈ slope * sqrt(m/z) + intercept, varies by charge
    # These are approximate values based on typical peptide behavior
    charge_params = {
        1: {'slope': 12.0, 'intercept': 150, 'color': '#e41a1c', 'label': 'z=1'},
        2: {'slope': 10.5, 'intercept': 180, 'color': '#377eb8', 'label': 'z=2'},
        3: {'slope': 9.0, 'intercept': 210, 'color': '#4daf4a', 'label': 'z=3'},
        4: {'slope': 7.5, 'intercept': 240, 'color': '#984ea3', 'label': 'z=4'},
    }

    def draw_plots(std_val):
        """Draw the plots on the current figure."""
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        # Left plot: CCS vs m/z for each charge
        for z, params in charge_params.items():
            ccs_mean = params['slope'] * np.sqrt(mz_range) + params['intercept']
            ccs_std = std_val * 100 * z  # Approximate spread scaling
            ax1.plot(mz_range, ccs_mean, color=params['color'], label=params['label'], linewidth=2)
            ax1.fill_between(mz_range, ccs_mean - ccs_std, ccs_mean + ccs_std,
                           color=params['color'], alpha=0.2)

        ax1.set_xlabel("m/z")
        ax1.set_ylabel("CCS (Å²)")
        ax1.set_title("CCS vs m/z by Charge State")
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Right plot: Inverse mobility distribution for a fixed m/z
        mz_fixed = 750
        x_im = np.linspace(0.6, 1.6, 200)

        for z, params in charge_params.items():
            ccs_mean = params['slope'] * np.sqrt(mz_fixed) + params['intercept']
            im_mean = 0.5 + (ccs_mean / 1000) * (z / 2)
            im_std = std_val * (1 + 0.2 * z)

            y = np.exp(-0.5 * ((x_im - im_mean) / im_std) ** 2)
            y = y / (y.max() + 1e-10) * z
            ax2.plot(x_im, y, color=params['color'], label=params['label'], linewidth=2)
            ax2.fill_between(x_im, 0, y, color=params['color'], alpha=0.3)

        ax2.set_xlabel("Inverse Ion Mobility (1/K₀, Vs/cm²)")
        ax2.set_ylabel("Relative Abundance")
        ax2.set_title(f"IM Distribution at m/z={mz_fixed}")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    with ui.card().classes("w-full"):
        ui.label("CCS Distribution by Charge State").classes("font-bold text-lg mb-2")
        ui.label("Shows how collision cross section varies with m/z for different charge states").classes("text-sm text-gray-600 dark:text-gray-400 mb-2")

        # Create plot using context manager
        with ui.pyplot(figsize=(10, 4)) as plot_element:
            draw_plots(inv_mob_std)

        def update_plot():
            """Update the plot with current parameters."""
            with plot_element:
                plt.clf()
                draw_plots(current_std[0])

        with ui.row().classes("w-full mt-4"):
            with ui.column().classes("flex-1"):
                ui.label("Inverse Mobility Std Dev Mean (1/K₀)")

                def handle_std_change(e):
                    current_std[0] = e.value
                    update_plot()
                    # Also call external callback if provided
                    if on_std_change is not None:
                        on_std_change(e.value)

                std_slider = ui.slider(
                    value=inv_mob_std,
                    min=0.001,
                    max=0.03,
                    step=0.001,
                    on_change=handle_std_change,
                ).classes("w-full")
                ui.label().bind_text_from(std_slider, 'value', lambda v: f"σ = {v:.4f} Vs/cm²")

    return {
        "plot": plot_element,
        "std_slider": std_slider,
        "update": update_plot,
    }


def create_acquisition_mode_plot(
    mode: str = "dia-PASEF",
    reference_path: str = None,
    show_precursor_data: bool = True,
) -> dict:
    """Create a visualization of acquisition mode isolation windows.

    Shows the selection rectangles in m/z vs ion mobility space for different
    acquisition modes to help users understand what each mode means.

    Args:
        mode: Acquisition mode (DDA, dia-PASEF, slice-PASEF, synchro-PASEF, midia-PASEF)
        reference_path: Path to reference .d dataset for accurate scan-to-mobility conversion
        show_precursor_data: Whether to show precursor data points from reference dataset

    Returns:
        Dictionary with plot element and update function
    """
    try:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyBboxPatch
        from matplotlib.collections import PatchCollection
    except ImportError:
        ui.label("Matplotlib not available for plotting")
        return {}

    current_mode = [mode]
    current_ref_path = [reference_path]
    current_show_precursors = [show_precursor_data]

    # Define the m/z and mobility ranges
    mz_min, mz_max = 400, 1200
    mob_min, mob_max = 0.6, 1.4

    # Try to get calibrated converter from reference dataset
    calibrated_converter = get_scan_mobility_converter(reference_path) if reference_path else None

    # Try to get precursor data from reference dataset
    precursor_data = get_precursor_data(reference_path) if reference_path and show_precursor_data else None

    def scan_to_mobility(scan, scan_max=962):
        """Convert scan number to 1/K0 value using calibration or approximation."""
        scan_arr = np.atleast_1d(np.array(scan))

        if calibrated_converter is not None:
            try:
                result = calibrated_converter(scan_arr)
                return result[0] if np.isscalar(scan) else result
            except Exception:
                pass

        # Fallback: approximate conversion
        # In Bruker timsTOF: low scan = high 1/K0 (heavy ions first)
        result = mob_max - (scan_arr / scan_max) * (mob_max - mob_min)
        return result[0] if np.isscalar(scan) else result

    # CCS model parameters (same as in create_ccs_charge_plot)
    charge_params = {
        1: {'slope': 12.0, 'intercept': 150, 'color': '#e41a1c'},
        2: {'slope': 10.5, 'intercept': 180, 'color': '#377eb8'},
        3: {'slope': 9.0, 'intercept': 210, 'color': '#4daf4a'},
        4: {'slope': 7.5, 'intercept': 240, 'color': '#984ea3'},
    }

    def mz_to_mobility(mz, charge, std=0.009):
        """Convert m/z to ion mobility using CCS model."""
        params = charge_params[charge]
        ccs = params['slope'] * np.sqrt(mz) + params['intercept']
        # Convert CCS to 1/K0 (approximate)
        im_mean = 0.5 + (ccs / 1000) * (charge / 2)
        # Add some noise
        im_std = std * (1 + 0.2 * charge)
        return im_mean + np.random.normal(0, im_std, size=np.array(mz).shape)

    def draw_acquisition_pattern(ax, acq_mode: str, precursors: dict = None):
        """Draw the isolation window pattern for a given acquisition mode."""
        ax.set_xlabel("m/z", fontsize=10, color='white')
        ax.set_ylabel("Inverse Mobility (1/K₀)", fontsize=10, color='white')
        ax.set_facecolor('#1a1a2e')
        # Make tick labels light
        ax.tick_params(colors='white', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        patches = []
        colors = []
        # Track bounds for auto-scaling
        all_mz = []
        all_mob = []

        # Store precursors for drawing after axis limits are set (so they get clipped to window bounds)
        has_precursors = precursors is not None and len(precursors.get('mz', [])) > 0

        if acq_mode == "DDA":
            # DDA: Individual precursor selections (narrow windows)
            ax.set_title("DDA - Data-Dependent Acquisition", fontsize=11, color='white')
            # Show example precursor selections spread across the space
            np.random.seed(123)
            for i in range(12):
                mz_center = 450 + i * 60
                # Approximate diagonal distribution (high m/z -> high 1/K0)
                mob_center = 0.7 + (mz_center - 400) / 800 * 0.5
                mob_center += np.random.uniform(-0.05, 0.05)
                width = 2  # narrow isolation window ~2 Da
                height = 0.03
                rect = Rectangle((mz_center - width/2, mob_center - height/2), width, height)
                patches.append(rect)
                colors.append('#ff6b6b')
                all_mz.extend([mz_center - width/2, mz_center + width/2])
                all_mob.extend([mob_center - height/2, mob_center + height/2])

        elif acq_mode == "dia-PASEF":
            # dia-PASEF: Load all windows from actual dia_ms_ms_windows.csv
            ax.set_title("dia-PASEF - Standard DIA", fontsize=11, color='white')
            window_df = load_window_config("dia_ms_ms_windows.csv")
            if window_df is not None:
                for _, row in window_df.iterrows():
                    mz_center = row['isolation_mz']
                    mz_width = row['isolation_width']
                    scan_start = int(row['scan_start'])
                    scan_end = int(row['scan_end'])
                    mob_start = scan_to_mobility(scan_end)
                    mob_end = scan_to_mobility(scan_start)
                    rect = FancyBboxPatch(
                        (mz_center - mz_width/2, mob_start), mz_width, mob_end - mob_start,
                        boxstyle="round,pad=0.01,rounding_size=0.01"
                    )
                    patches.append(rect)
                    colors.append('#4ecdc4')
                    all_mz.extend([mz_center - mz_width/2, mz_center + mz_width/2])
                    all_mob.extend([mob_start, mob_end])

        elif acq_mode == "slice-PASEF":
            # slice-PASEF: Load all windows from actual slice_ms_ms_windows.csv
            ax.set_title("slice-PASEF - Mobility Slices", fontsize=11, color='white')
            window_df = load_window_config("slice_ms_ms_windows.csv")
            if window_df is not None:
                # Color by window group for visual distinction
                max_group = int(window_df['window_group'].max())
                for _, row in window_df.iterrows():
                    mz_center = row['isolation_mz']
                    mz_width = row['isolation_width']
                    scan_start = int(row['scan_start'])
                    scan_end = int(row['scan_end'])
                    group = int(row['window_group'])
                    mob_start = scan_to_mobility(scan_end)
                    mob_end = scan_to_mobility(scan_start)
                    rect = FancyBboxPatch(
                        (mz_center - mz_width/2, mob_start), mz_width, mob_end - mob_start,
                        boxstyle="round,pad=0.01,rounding_size=0.01"
                    )
                    patches.append(rect)
                    colors.append(plt.cm.viridis(group / max_group))
                    all_mz.extend([mz_center - mz_width/2, mz_center + mz_width/2])
                    all_mob.extend([mob_start, mob_end])

        elif acq_mode == "synchro-PASEF":
            # synchro-PASEF: Load windows from actual synchro_ms_ms_windows.csv
            window_df = load_window_config("synchro_ms_ms_windows.csv")
            if window_df is not None:
                total_windows = len(window_df)
                # Show all windows if under 1000, otherwise subsample
                if total_windows <= 1000:
                    sampled_df = window_df
                else:
                    sample_step = max(1, total_windows // 1000)
                    sampled_df = window_df.iloc[::sample_step]
                ax.set_title(f"synchro-PASEF (showing {len(sampled_df)} of {total_windows} windows)", fontsize=10, color='white')
                max_scan = int(window_df['scan_end'].max())
                for _, row in sampled_df.iterrows():
                    mz_center = row['isolation_mz']
                    mz_width = row['isolation_width']
                    scan_start = int(row['scan_start'])
                    scan_end = int(row['scan_end'])
                    mob_start = scan_to_mobility(scan_end)
                    mob_end = scan_to_mobility(scan_start)
                    # Color by scan position for gradient effect
                    t = scan_start / max_scan if max_scan > 0 else 0
                    rect = FancyBboxPatch(
                        (mz_center - mz_width/2, mob_start), mz_width, mob_end - mob_start,
                        boxstyle="round,pad=0.005,rounding_size=0.005"
                    )
                    patches.append(rect)
                    colors.append(plt.cm.plasma(t))
                    all_mz.extend([mz_center - mz_width/2, mz_center + mz_width/2])
                    all_mob.extend([mob_start, mob_end])

        elif acq_mode == "midia-PASEF":
            # midia-PASEF: Load windows from actual midia_ms_ms_windows.csv
            window_df = load_window_config("midia_ms_ms_windows.csv")
            if window_df is not None:
                total_windows = len(window_df)
                # Show all windows if under 2000, otherwise subsample
                if total_windows <= 2000:
                    sampled_df = window_df
                else:
                    sample_step = max(1, total_windows // 2000)
                    sampled_df = window_df.iloc[::sample_step]
                ax.set_title(f"midia-PASEF (showing {len(sampled_df)} of {total_windows} windows)", fontsize=10, color='white')
                # Color by window group for multiplexed streams
                group_colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#a855f7', '#22c55e']
                for _, row in sampled_df.iterrows():
                    mz_center = row['isolation_mz']
                    mz_width = row['isolation_width']
                    scan_start = int(row['scan_start'])
                    scan_end = int(row['scan_end'])
                    group = int(row['window_group'])
                    mob_start = scan_to_mobility(scan_end)
                    mob_end = scan_to_mobility(scan_start)
                    rect = FancyBboxPatch(
                        (mz_center - mz_width/2, mob_start), mz_width, mob_end - mob_start,
                        boxstyle="round,pad=0.005,rounding_size=0.005"
                    )
                    patches.append(rect)
                    colors.append(group_colors[(group - 1) % len(group_colors)])
                    all_mz.extend([mz_center - mz_width/2, mz_center + mz_width/2])
                    all_mob.extend([mob_start, mob_end])

        # Set axis limits based on window bounds with padding
        if all_mz and all_mob:
            mz_range = max(all_mz) - min(all_mz)
            mob_range = max(all_mob) - min(all_mob)
            # Use 10% padding to ensure all windows are fully visible
            padding_mz = mz_range * 0.10
            padding_mob = mob_range * 0.10
            xlim = (min(all_mz) - padding_mz, max(all_mz) + padding_mz)
            ylim = (min(all_mob) - padding_mob, max(all_mob) + padding_mob)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            xlim = (mz_min, mz_max)
            ylim = (mob_min, mob_max)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        # Draw precursor data points (clipped to axis limits)
        if has_precursors:
            # Filter precursor data to only show points within the window bounds
            mz_mask = (precursors['mz'] >= xlim[0]) & (precursors['mz'] <= xlim[1])
            mob_mask = (precursors['mobility'] >= ylim[0]) & (precursors['mobility'] <= ylim[1])
            mask = mz_mask & mob_mask
            if np.any(mask):
                ax.scatter(
                    precursors['mz'][mask],
                    precursors['mobility'][mask],
                    s=1,  # Small point size
                    c='#888888',  # Gray color
                    alpha=0.4,
                    marker='.',
                    rasterized=True,  # Better performance for many points
                    zorder=1,  # Draw behind windows
                )

        # Add patches to axes (drawn on top of precursors)
        if patches:
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                patch.set_edgecolor('white')
                patch.set_linewidth(0.5)
                patch.set_zorder(2)  # Draw on top of precursors
                ax.add_patch(patch)

        ax.grid(True, alpha=0.2, color='white')

    # Create the plot
    with ui.pyplot(figsize=(6, 3.5), close=False) as plot_element:
        fig = plt.gcf()
        fig.patch.set_facecolor('#0d1117')
        ax = plt.subplot(1, 1, 1)
        draw_acquisition_pattern(ax, mode, precursor_data if current_show_precursors[0] else None)
        plt.tight_layout()

    def update_plot(new_mode: str, new_ref_path: str = None, new_show_precursors: bool = None):
        """Update the plot with a new acquisition mode and optionally new reference path."""
        nonlocal calibrated_converter, precursor_data

        current_mode[0] = new_mode

        # Update show precursors setting
        if new_show_precursors is not None:
            current_show_precursors[0] = new_show_precursors

        # Update calibration and precursor data if reference path changed
        if new_ref_path is not None and new_ref_path != current_ref_path[0]:
            current_ref_path[0] = new_ref_path
            calibrated_converter = get_scan_mobility_converter(new_ref_path) if new_ref_path else None
            precursor_data = get_precursor_data(new_ref_path) if new_ref_path and current_show_precursors[0] else None

        with plot_element:
            plt.clf()
            fig = plt.gcf()
            fig.patch.set_facecolor('#0d1117')
            ax = plt.subplot(1, 1, 1)
            draw_acquisition_pattern(ax, new_mode, precursor_data if current_show_precursors[0] else None)
            plt.tight_layout()

    def set_reference(new_ref_path: str):
        """Update the reference path for calibration."""
        nonlocal calibrated_converter, precursor_data
        current_ref_path[0] = new_ref_path
        calibrated_converter = get_scan_mobility_converter(new_ref_path) if new_ref_path else None
        precursor_data = get_precursor_data(new_ref_path) if new_ref_path and current_show_precursors[0] else None
        # Redraw with current mode
        update_plot(current_mode[0])

    def toggle_precursors(show: bool):
        """Toggle showing precursor data points."""
        nonlocal precursor_data
        current_show_precursors[0] = show
        if show and current_ref_path[0] and precursor_data is None:
            precursor_data = get_precursor_data(current_ref_path[0])
        update_plot(current_mode[0])

    def get_has_calibration():
        """Check if calibration is currently available."""
        return calibrated_converter is not None

    result = {
        "plot": plot_element,
        "update": update_plot,
        "set_reference": set_reference,
        "toggle_precursors": toggle_precursors,
    }

    # Make has_calibration a dynamic property
    class CalibrationStatus:
        def __bool__(self):
            return calibrated_converter is not None

    class HasPrecursorData:
        def __bool__(self):
            return precursor_data is not None and len(precursor_data.get('mz', [])) > 0

    result["has_calibration"] = CalibrationStatus()
    result["has_precursor_data"] = HasPrecursorData()

    return result


def create_simple_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> Optional[ui.pyplot]:
    """Create a simple bar chart.

    Args:
        labels: Bar labels
        values: Bar values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        The pyplot element or None if matplotlib not available
    """
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    return ui.pyplot(fig)
