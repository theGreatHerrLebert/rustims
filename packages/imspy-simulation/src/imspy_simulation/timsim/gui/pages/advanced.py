"""Advanced tabbed interface for TimSim GUI."""

import asyncio
from typing import Callable, TYPE_CHECKING
from nicegui import ui

from ..components.file_inputs import fasta_picker, reference_picker, output_picker, path_input, FileBrowser
from ..components.config_sections import (
    ConfigSection,
    number_field,
    slider_field,
    toggle_field,
    select_field,
    text_field,
)
from ..components.console import LogConsole, create_progress_section
from ..components.plots import create_emg_plot, create_ccs_charge_plot, create_acquisition_mode_plot
from ..components.visualization3d import create_3d_visualization

if TYPE_CHECKING:
    from ..state import SimulationState


def create_advanced_view(
    state: "SimulationState",
    on_run: Callable[[], None],
    on_stop: Callable[[], None],
) -> ui.element:
    """Create the advanced tabbed interface.

    Args:
        state: Shared simulation state
        on_run: Callback to start simulation
        on_stop: Callback to stop simulation

    Returns:
        The container element
    """
    with ui.column().classes("w-full") as container:
        # Dark mode (default to dark)
        dark = ui.dark_mode(True)

        # Header
        with ui.row().classes("w-full items-center justify-between mb-4"):
            ui.label("TimSim - Simulation Configuration").classes("text-2xl font-bold")
            with ui.row().classes("gap-2"):
                # Dark mode toggle
                def toggle_dark():
                    dark.toggle()

                ui.button(
                    icon="dark_mode",
                    on_click=toggle_dark,
                ).props("flat round").tooltip("Toggle dark mode")

                ui.button(
                    "Load Config",
                    icon="upload",
                    on_click=lambda: asyncio.create_task(load_config_dialog(state)),
                ).props("flat")

                def download_config():
                    """Download config as TOML file."""
                    import tempfile
                    from pathlib import Path

                    # Generate filename
                    name = state.config.experiment.experiment_name.replace(
                        "[PLACEHOLDER]", state.config.experiment.acquisition_type
                    )
                    name = name.replace(" ", "_").replace("/", "_")
                    filename = f"{name}_config.toml"

                    # Save to temp file and trigger download
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                        state.config.save_toml(f.name)
                        ui.download(f.name, filename)

                ui.button(
                    "Download Config",
                    icon="download",
                    on_click=download_config,
                ).props("flat")

        # Main content with tabs
        with ui.splitter(value=70).classes("w-full").style("height: calc(100vh - 100px)") as splitter:
            # Left: Configuration tabs
            with splitter.before:
                with ui.tabs().classes("w-full") as tabs:
                    files_tab = ui.tab("Files", icon="folder")
                    digestion_tab = ui.tab("Digestion", icon="content_cut")
                    chromatography_tab = ui.tab("Chromatography", icon="science")
                    ions_tab = ui.tab("Ions", icon="show_chart")
                    noise_tab = ui.tab("Noise", icon="grain")
                    acquisition_tab = ui.tab("Acquisition", icon="settings")
                    performance_tab = ui.tab("Performance", icon="speed")
                    preview_3d_tab = ui.tab("3D Preview", icon="view_in_ar")
                    help_tab = ui.tab("Help", icon="help")

                with ui.tab_panels(tabs, value=files_tab).classes("w-full"):
                    # Files Tab
                    with ui.tab_panel(files_tab):
                        create_files_panel(state)

                    # Digestion Tab
                    with ui.tab_panel(digestion_tab):
                        create_digestion_panel(state)

                    # Chromatography Tab
                    with ui.tab_panel(chromatography_tab):
                        create_chromatography_panel(state)

                    # Ions Tab
                    with ui.tab_panel(ions_tab):
                        create_ions_panel(state)

                    # Noise Tab
                    with ui.tab_panel(noise_tab):
                        create_noise_panel(state)

                    # Acquisition Tab
                    with ui.tab_panel(acquisition_tab):
                        create_acquisition_panel(state)

                    # Performance Tab
                    with ui.tab_panel(performance_tab):
                        create_performance_panel(state)

                    # 3D Preview Tab
                    with ui.tab_panel(preview_3d_tab):
                        create_3d_preview_panel(state)

                    # Help Tab
                    with ui.tab_panel(help_tab):
                        create_help_panel()

            # Right: Console and Run
            with splitter.after:
                with ui.column().classes("w-full h-full p-4"):
                    # Progress section
                    progress_bar, status_label = create_progress_section()

                    # Console
                    console = LogConsole(max_lines=500)
                    console.create()

                    # Store references for external access
                    state._console = console
                    state._progress_bar = progress_bar
                    state._status_label = status_label

                    ui.space()

                    # Run/Stop button with toggle functionality
                    def toggle_run_button():
                        """Update button appearance based on running state."""
                        if state.is_running:
                            run_btn.text = "Stop Simulation"
                            run_btn._props["icon"] = "stop"
                            run_btn.props("color=negative")
                        else:
                            run_btn.text = "Start Simulation"
                            run_btn._props["icon"] = "play_arrow"
                            run_btn.props("color=positive")
                        run_btn.update()

                    def handle_run_click():
                        if state.is_running:
                            on_stop()
                        else:
                            on_run()
                        # Schedule button update after state changes
                        ui.timer(0.1, toggle_run_button, once=True)

                    run_btn = ui.button(
                        "Start Simulation",
                        icon="play_arrow",
                        on_click=handle_run_click,
                    ).classes("w-full h-14 text-lg")
                    run_btn.props("color=positive")

                    # Store toggle function for external updates
                    state._toggle_run_button = toggle_run_button

    return container


def create_files_panel(state: "SimulationState") -> None:
    """Create the Files configuration panel."""
    # References for acquisition plot and calibration label (populated later)
    acq_plot_ref = [None]
    calibration_label_ref = [None]

    def on_reference_change(path: str):
        """Handle reference path change - update config, acquisition plot, and 3D visualization."""
        setattr(state.config.paths, "reference_path", path)
        # Update acquisition plot if available
        if acq_plot_ref[0] and "set_reference" in acq_plot_ref[0]:
            acq_plot_ref[0]["set_reference"](path)
            # Update calibration status label
            if calibration_label_ref[0]:
                if acq_plot_ref[0].get("has_calibration"):
                    calibration_label_ref[0].text = "Using calibrated scan-to-mobility from reference dataset"
                    calibration_label_ref[0].classes(remove="text-gray-600", add="text-green-500")
                else:
                    calibration_label_ref[0].text = "No reference dataset - using approximate conversion"
                    calibration_label_ref[0].classes(remove="text-green-500", add="text-gray-600")
        # Update 3D visualization if available
        if hasattr(state, '_3d_vis_component') and state._3d_vis_component:
            state._3d_vis_component["set_data_path"](path)

    with ui.column().classes("w-full gap-4 p-4"):
        ui.label("Input/Output Paths").classes("text-lg font-bold")

        reference_picker(
            value=state.config.paths.reference_path,
            on_change=on_reference_change,
        )

        fasta_picker(
            value=state.config.paths.fasta_path,
            on_change=lambda v: setattr(state.config.paths, "fasta_path", v),
        )

        output_picker(
            value=state.config.paths.save_path,
            on_change=lambda v: setattr(state.config.paths, "save_path", v),
        )

        path_input(
            label="Modifications File (optional)",
            value=state.config.paths.modifications,
            on_change=lambda v: setattr(state.config.paths, "modifications", v),
            placeholder="Leave empty for defaults",
        )

        ui.separator()

        ui.label("Experiment Settings").classes("text-lg font-bold mt-4")

        text_field(
            label="Experiment Name",
            value=state.config.experiment.experiment_name,
            on_change=lambda v: setattr(state.config.experiment, "experiment_name", v),
            tooltip="Use [PLACEHOLDER] to auto-insert acquisition type",
        )

        pasef_variants = ["dia-PASEF", "slice-PASEF", "synchro-PASEF", "midia-PASEF"]

        # Determine initial values
        current_acq = state.config.experiment.acquisition_type
        is_dda = current_acq == "DDA"
        base_acq = "DDA" if is_dda else "DIA"
        pasef_value = current_acq if current_acq in pasef_variants else "dia-PASEF"
        initial_plot_mode = "DDA" if is_dda else pasef_value

        # Store plot reference for updates
        acq_plot_ref = [None]

        with ui.row().classes("w-full gap-4 items-end"):
            with ui.column().classes("flex-1"):
                def on_base_acq_change(v):
                    setattr(state.config.experiment, "acquisition_type", v)
                    # Enable/disable PASEF select based on mode
                    if v == "DDA":
                        pasef_select.disable()
                        # Update plot to show DDA pattern
                        if acq_plot_ref[0] and "update" in acq_plot_ref[0]:
                            acq_plot_ref[0]["update"]("DDA")
                    else:
                        pasef_select.enable()
                        # Set to dia-PASEF when switching back to DIA
                        if state.config.experiment.acquisition_type == "DIA":
                            pasef_select.value = "dia-PASEF"
                            setattr(state.config.experiment, "acquisition_type", "dia-PASEF")
                        # Update plot
                        if acq_plot_ref[0] and "update" in acq_plot_ref[0]:
                            acq_plot_ref[0]["update"](pasef_select.value)

                select_field(
                    label="Acquisition Type",
                    value=base_acq,
                    options=["DIA", "DDA"],
                    on_change=on_base_acq_change,
                )
            with ui.column().classes("flex-1"):
                ui.label("DIA-PASEF Variants").classes("text-xs text-gray-500 dark:text-gray-400 mb-1")

                def on_pasef_select(v):
                    if v:
                        setattr(state.config.experiment, "acquisition_type", v)
                        # Update plot
                        if acq_plot_ref[0] and "update" in acq_plot_ref[0]:
                            acq_plot_ref[0]["update"](v)

                pasef_select = ui.select(
                    options=pasef_variants,
                    value=pasef_value,
                    on_change=lambda e: on_pasef_select(e.value),
                ).classes("w-full").props('dense')

                # Disable if DDA is selected
                if is_dda:
                    pasef_select.disable()

        # Add acquisition mode visualization
        with ui.card().classes("w-full mt-4"):
            ui.label("Isolation Window Pattern").classes("font-bold text-lg mb-2")
            calibration_label_ref[0] = ui.label("").classes("text-sm text-gray-600 dark:text-gray-400 mb-2")
            acq_plot_ref[0] = create_acquisition_mode_plot(
                mode=initial_plot_mode,
                reference_path=state.config.paths.reference_path,
            )
            # Update calibration status label
            if acq_plot_ref[0].get("has_calibration"):
                calibration_label_ref[0].text = "Using calibrated scan-to-mobility from reference dataset"
                calibration_label_ref[0].classes(remove="text-gray-600", add="text-green-500")
            else:
                calibration_label_ref[0].text = "No reference dataset - using approximate conversion"
                calibration_label_ref[0].classes(remove="text-green-500", add="text-gray-600")

        number_field(
            label="Gradient Length",
            value=state.config.experiment.gradient_length,
            on_change=lambda v: setattr(state.config.experiment, "gradient_length", v),
            min_val=60,
            step=60,
            suffix="seconds",
            tooltip="Total LC gradient duration affecting peptide elution distribution",
        )

        toggle_field(
            label="Use Reference Layout",
            value=state.config.experiment.use_reference_layout,
            on_change=lambda v: setattr(state.config.experiment, "use_reference_layout", v),
            tooltip="Use frame/scan layout from reference dataset",
        )

        toggle_field(
            label="Load Reference in Memory",
            value=state.config.experiment.reference_in_memory,
            on_change=lambda v: setattr(state.config.experiment, "reference_in_memory", v),
            tooltip="Faster but uses more RAM",
        )

        toggle_field(
            label="Use Bruker SDK",
            value=state.config.experiment.use_bruker_sdk,
            on_change=lambda v: setattr(state.config.experiment, "use_bruker_sdk", v),
            tooltip="Required for accurate mass/mobility calibration (not available on macOS)",
        )

        toggle_field(
            label="Apply Fragmentation",
            value=state.config.experiment.apply_fragmentation,
            on_change=lambda v: setattr(state.config.experiment, "apply_fragmentation", v),
            tooltip="Generate fragment ions for MS2 spectra",
        )


def create_digestion_panel(state: "SimulationState") -> None:
    """Create the Digestion configuration panel."""
    with ui.column().classes("w-full gap-4 p-4"):
        ui.label("Protein & Peptide Settings").classes("text-lg font-bold")

        number_field(
            label="Number of Proteins",
            value=state.config.digestion.n_proteins,
            on_change=lambda v: setattr(state.config.digestion, "n_proteins", int(v)),
            min_val=1,
            step=1000,
            tooltip="Maximum number of proteins to load from FASTA file",
        )

        number_field(
            label="Total Peptides (before sampling)",
            value=state.config.digestion.num_peptides_total,
            on_change=lambda v: setattr(state.config.digestion, "num_peptides_total", int(v)),
            min_val=1,
            step=10000,
            tooltip="Total peptides to generate from digestion before sampling",
        )

        number_field(
            label="Sample Peptides",
            value=state.config.digestion.num_sample_peptides,
            on_change=lambda v: setattr(state.config.digestion, "num_sample_peptides", int(v)),
            min_val=1,
            step=1000,
            tooltip="Number of peptides to randomly sample for simulation",
        )

        toggle_field(
            label="Sample Peptides",
            value=state.config.digestion.sample_peptides,
            on_change=lambda v: setattr(state.config.digestion, "sample_peptides", v),
            tooltip="Enable random sampling of peptides for faster simulation",
        )

        number_field(
            label="Random Seed",
            value=state.config.digestion.sample_seed,
            on_change=lambda v: setattr(state.config.digestion, "sample_seed", int(v)),
            min_val=0,
            step=1,
            tooltip="Seed for reproducible random sampling (0 = random)",
        )

        ui.separator()

        ui.label("Enzyme Settings").classes("text-lg font-bold mt-4")

        toggle_field(
            label="Digest Proteins",
            value=state.config.digestion.digest_proteins,
            on_change=lambda v: setattr(state.config.digestion, "digest_proteins", v),
            tooltip="Perform in-silico enzymatic digestion of proteins",
        )

        text_field(
            label="Cleavage Sites",
            value=state.config.digestion.cleave_at,
            on_change=lambda v: setattr(state.config.digestion, "cleave_at", v),
            tooltip="Amino acids where enzyme cleaves (e.g., KR for trypsin)",
        )

        text_field(
            label="Restrict Cleavage",
            value=state.config.digestion.restrict,
            on_change=lambda v: setattr(state.config.digestion, "restrict", v),
            tooltip="Amino acids that block cleavage (e.g., P for proline)",
        )

        number_field(
            label="Missed Cleavages",
            value=state.config.digestion.missed_cleavages,
            on_change=lambda v: setattr(state.config.digestion, "missed_cleavages", int(v)),
            min_val=0,
            max_val=5,
            step=1,
            tooltip="Number of allowed missed enzymatic cleavage sites",
        )

        with ui.row().classes("w-full gap-4"):
            number_field(
                label="Min Length",
                value=state.config.digestion.min_len,
                on_change=lambda v: setattr(state.config.digestion, "min_len", int(v)),
                min_val=1,
                step=1,
                tooltip="Minimum peptide length in amino acids",
            )

            number_field(
                label="Max Length",
                value=state.config.digestion.max_len,
                on_change=lambda v: setattr(state.config.digestion, "max_len", int(v)),
                min_val=1,
                step=1,
                tooltip="Maximum peptide length in amino acids",
            )

        toggle_field(
            label="Generate Decoys",
            value=state.config.digestion.decoys,
            on_change=lambda v: setattr(state.config.digestion, "decoys", v),
            tooltip="Generate reversed decoy peptides for FDR estimation",
        )

        toggle_field(
            label="Remove Degenerate Peptides",
            value=state.config.digestion.remove_degenerate_peptides,
            on_change=lambda v: setattr(state.config.digestion, "remove_degenerate_peptides", v),
            tooltip="Remove peptides that map to multiple proteins",
        )

        number_field(
            label="Intensity Upscale Factor",
            value=state.config.digestion.upscale_factor,
            on_change=lambda v: setattr(state.config.digestion, "upscale_factor", int(v)),
            min_val=1,
            step=10000,
            tooltip="Factor to scale up peptide intensities",
        )


def create_chromatography_panel(state: "SimulationState") -> None:
    """Create the Chromatography configuration panel."""
    with ui.column().classes("w-full gap-4 p-4"):
        ui.label("Retention Time Settings").classes("text-lg font-bold")

        # RT Model selection - supports local PyTorch or Koina models
        rt_model_options = [
            "Local PyTorch (default)",
            "Deeplc_hela_hf (Koina)",
            "Chronologer_RT (Koina)",
            "AlphaPeptDeep_rt_generic (Koina)",
            "Prosit_2019_irt (Koina)",
        ]

        # Map current value to display name
        current_rt_model = state.config.models.rt_model or state.config.retention_time.koina_rt_model
        rt_display_value = "Local PyTorch (default)"
        if current_rt_model:
            for opt in rt_model_options:
                if current_rt_model in opt:
                    rt_display_value = opt
                    break

        def on_rt_model_change(v):
            # Extract model name from display string
            if "Local" in v or "default" in v:
                model_name = ""
            else:
                # Extract the actual model name (before " (Koina)")
                model_name = v.split(" (Koina)")[0] if "(Koina)" in v else v
            setattr(state.config.models, "rt_model", model_name)
            # Also set legacy field for backwards compatibility
            setattr(state.config.retention_time, "koina_rt_model", model_name)

        select_field(
            label="RT Prediction Model",
            value=rt_display_value,
            options=rt_model_options,
            on_change=on_rt_model_change,
            tooltip="Deep learning model for retention time prediction. Local = PyTorch model, Koina = online API",
        )

        number_field(
            label="Min RT Percent",
            value=state.config.retention_time.min_rt_percent,
            on_change=lambda v: setattr(state.config.retention_time, "min_rt_percent", v),
            min_val=0,
            max_val=50,
            step=0.5,
            suffix="%",
            tooltip="Minimum retention time as percentage of gradient",
        )

        toggle_field(
            label="Exclude Accumulated Gradient Start",
            value=state.config.retention_time.exclude_accumulated_gradient_start,
            on_change=lambda v: setattr(
                state.config.retention_time, "exclude_accumulated_gradient_start", v
            ),
            tooltip="Skip the accumulation phase at the start of the gradient",
        )

        ui.separator()

        ui.label("EMG Peak Shape Parameters").classes("text-lg font-bold mt-4")

        # Sigma parameters
        with ui.expansion("Sigma (Peak Width)", icon="straighten"):
            with ui.row().classes("w-full gap-4"):
                number_field(
                    label="Lower",
                    value=state.config.retention_time.sigma_lower_rt,
                    on_change=lambda v: setattr(state.config.retention_time, "sigma_lower_rt", v),
                    min_val=0.1,
                    step=0.1,
                )
                number_field(
                    label="Upper",
                    value=state.config.retention_time.sigma_upper_rt,
                    on_change=lambda v: setattr(state.config.retention_time, "sigma_upper_rt", v),
                    min_val=0.1,
                    step=0.1,
                )
            with ui.row().classes("w-full gap-4"):
                number_field(
                    label="Alpha",
                    value=state.config.retention_time.sigma_alpha_rt,
                    on_change=lambda v: setattr(state.config.retention_time, "sigma_alpha_rt", v),
                    min_val=0.1,
                    step=0.5,
                )
                number_field(
                    label="Beta",
                    value=state.config.retention_time.sigma_beta_rt,
                    on_change=lambda v: setattr(state.config.retention_time, "sigma_beta_rt", v),
                    min_val=0.1,
                    step=0.5,
                )

        # K parameters
        with ui.expansion("K (Tailing)", icon="trending_up"):
            with ui.row().classes("w-full gap-4"):
                number_field(
                    label="Lower",
                    value=state.config.retention_time.k_lower_rt,
                    on_change=lambda v: setattr(state.config.retention_time, "k_lower_rt", v),
                    min_val=0,
                    step=0.5,
                )
                number_field(
                    label="Upper",
                    value=state.config.retention_time.k_upper_rt,
                    on_change=lambda v: setattr(state.config.retention_time, "k_upper_rt", v),
                    min_val=0,
                    step=0.5,
                )
            with ui.row().classes("w-full gap-4"):
                number_field(
                    label="Alpha",
                    value=state.config.retention_time.k_alpha_rt,
                    on_change=lambda v: setattr(state.config.retention_time, "k_alpha_rt", v),
                    min_val=0.1,
                    step=0.5,
                )
                number_field(
                    label="Beta",
                    value=state.config.retention_time.k_beta_rt,
                    on_change=lambda v: setattr(state.config.retention_time, "k_beta_rt", v),
                    min_val=0.1,
                    step=0.5,
                )

        # EMG Preview
        with ui.expansion("Peak Shape Preview", icon="preview", value=False):
            create_emg_plot(
                sigma=(state.config.retention_time.sigma_lower_rt + state.config.retention_time.sigma_upper_rt) / 2,
                k=(state.config.retention_time.k_lower_rt + state.config.retention_time.k_upper_rt) / 2,
            )

        ui.separator()

        ui.label("Ion Mobility / CCS").classes("text-lg font-bold mt-4")

        # CCS Model selection - supports local PyTorch or Koina models
        ccs_model_options = [
            "Local PyTorch (default)",
            "AlphaPeptDeep_ccs_generic (Koina)",
            "IM2Deep (Koina)",
        ]

        # Map current value to display name
        current_ccs_model = state.config.models.ccs_model or ""
        ccs_display_value = "Local PyTorch (default)"
        if current_ccs_model:
            for opt in ccs_model_options:
                if current_ccs_model in opt:
                    ccs_display_value = opt
                    break

        def on_ccs_model_change(v):
            # Extract model name from display string
            if "Local" in v or "default" in v:
                model_name = ""
            else:
                # Extract the actual model name (before " (Koina)")
                model_name = v.split(" (Koina)")[0] if "(Koina)" in v else v
            setattr(state.config.models, "ccs_model", model_name)

        select_field(
            label="CCS Prediction Model",
            value=ccs_display_value,
            options=ccs_model_options,
            on_change=on_ccs_model_change,
            tooltip="Model for CCS/ion mobility prediction. Local = PyTorch, Koina = online API. AlphaPeptDeep supports phospho modifications.",
        )

        toggle_field(
            label="Use Mean Inverse Mobility Std",
            value=state.config.ion_mobility.use_inverse_mobility_std_mean,
            on_change=lambda v: setattr(
                state.config.ion_mobility, "use_inverse_mobility_std_mean", v
            ),
        )

        # CCS/IM visualization with integrated slider that updates config
        with ui.expansion("CCS/Ion Mobility Distribution", icon="show_chart", value=True):
            create_ccs_charge_plot(
                inv_mob_std=state.config.ion_mobility.inverse_mobility_std_mean,
                on_std_change=lambda v: setattr(
                    state.config.ion_mobility, "inverse_mobility_std_mean", v
                ),
            )


def create_ions_panel(state: "SimulationState") -> None:
    """Create the Ions configuration panel."""
    with ui.column().classes("w-full gap-4 p-4"):
        ui.label("Charge State Settings").classes("text-lg font-bold")

        number_field(
            label="Charge Probability (p)",
            value=state.config.charge_states.p_charge,
            on_change=lambda v: setattr(state.config.charge_states, "p_charge", v),
            min_val=0,
            max_val=1,
            step=0.05,
        )

        number_field(
            label="Maximum Charge State",
            value=state.config.charge_states.max_charge,
            on_change=lambda v: setattr(state.config.charge_states, "max_charge", int(v)),
            min_val=1,
            max_val=6,
            step=1,
        )

        number_field(
            label="Min Charge Contribution",
            value=state.config.charge_states.min_charge_contrib,
            on_change=lambda v: setattr(state.config.charge_states, "min_charge_contrib", v),
            min_val=0,
            max_val=1,
            step=0.005,
        )

        toggle_field(
            label="Use Binomial Charge Model",
            value=state.config.charge_states.binomial_charge_model,
            on_change=lambda v: setattr(state.config.charge_states, "binomial_charge_model", v),
            tooltip="Use binomial model instead of deep learning model",
        )

        toggle_field(
            label="Normalize Charge States",
            value=state.config.charge_states.normalize_charge_states,
            on_change=lambda v: setattr(state.config.charge_states, "normalize_charge_states", v),
        )

        number_field(
            label="Charge +1 Probability",
            value=state.config.charge_states.charge_state_one_probability,
            on_change=lambda v: setattr(
                state.config.charge_states, "charge_state_one_probability", v
            ),
            min_val=0,
            max_val=1,
            step=0.05,
        )

        ui.separator()

        ui.label("Isotopic Pattern").classes("text-lg font-bold mt-4")

        number_field(
            label="Number of Isotope Peaks",
            value=state.config.isotopic_pattern.isotope_k,
            on_change=lambda v: setattr(state.config.isotopic_pattern, "isotope_k", int(v)),
            min_val=1,
            max_val=15,
            step=1,
        )

        number_field(
            label="Min Isotope Intensity",
            value=state.config.isotopic_pattern.isotope_min_intensity,
            on_change=lambda v: setattr(
                state.config.isotopic_pattern, "isotope_min_intensity", int(v)
            ),
            min_val=0,
            step=1,
        )

        toggle_field(
            label="Centroid Isotopes",
            value=state.config.isotopic_pattern.isotope_centroid,
            on_change=lambda v: setattr(state.config.isotopic_pattern, "isotope_centroid", v),
        )

        ui.separator()

        ui.label("Fragment Intensity").classes("text-lg font-bold mt-4")

        # Fragment Intensity Model selection - local PyTorch or Koina models
        intensity_model_options = [
            "Local PyTorch (PROSPECT, default)",
            "Prosit (Koina)",
            "AlphaPeptDeep (Koina, phospho-compatible)",
            "MS2PIP (Koina)",
        ]

        # Map current value to display name
        current_intensity_model = state.config.models.intensity_model or state.config.fragment_intensity.fragment_intensity_model
        intensity_display_value = "Local PyTorch (PROSPECT, default)"
        if current_intensity_model:
            model_lower = current_intensity_model.lower()
            if "alphapeptdeep" in model_lower:
                intensity_display_value = "AlphaPeptDeep (Koina, phospho-compatible)"
            elif "ms2pip" in model_lower:
                intensity_display_value = "MS2PIP (Koina)"
            elif "prosit" in model_lower:
                intensity_display_value = "Prosit (Koina)"
            elif model_lower in ("", "local"):
                intensity_display_value = "Local PyTorch (PROSPECT, default)"

        def on_intensity_model_change(v):
            # Map display name to model key
            if "AlphaPeptDeep" in v:
                model_name = "alphapeptdeep"
            elif "MS2PIP" in v:
                model_name = "ms2pip"
            elif "Prosit" in v:
                model_name = "prosit"
            else:
                model_name = ""  # Default to local PyTorch
            setattr(state.config.models, "intensity_model", model_name)
            # Also set legacy field for backwards compatibility
            setattr(state.config.fragment_intensity, "fragment_intensity_model", model_name)

        select_field(
            label="Fragment Intensity Model",
            value=intensity_display_value,
            options=intensity_model_options,
            on_change=on_intensity_model_change,
            tooltip="Model for MS2 fragment intensity prediction. Local uses PROSPECT fine-tuned PyTorch model. AlphaPeptDeep (Koina) supports phosphorylation.",
        )

        # Show phospho compatibility warning
        ui.label(
            "Note: For phosphorylated peptides, use AlphaPeptDeep (Koina) which supports all modifications."
        ).classes("text-xs text-amber-500 dark:text-amber-400 mt-1")

        number_field(
            label="Downsampling Factor",
            value=state.config.fragment_intensity.down_sample_factor,
            on_change=lambda v: setattr(
                state.config.fragment_intensity, "down_sample_factor", v
            ),
            min_val=0,
            max_val=1,
            step=0.1,
        )

        ui.separator()

        ui.label("Quadrupole Transmission").classes("text-lg font-bold mt-4")

        select_field(
            label="Isotope Transmission Mode",
            value=state.config.quad_transmission.quad_isotope_transmission_mode,
            options=["none", "precursor_scaling", "per_fragment"],
            on_change=lambda v: setattr(
                state.config.quad_transmission,
                "quad_isotope_transmission_mode",
                v,
            ),
            tooltip="Mode for quad-selection dependent isotope transmission. "
                    "'none' = disabled, 'precursor_scaling' = scale based on precursor, "
                    "'per_fragment' = accurate mode with individual fragment ion recalculation",
        )

        number_field(
            label="Min Transmission Probability",
            value=state.config.quad_transmission.quad_transmission_min_probability,
            on_change=lambda v: setattr(
                state.config.quad_transmission, "quad_transmission_min_probability", v
            ),
            min_val=0,
            max_val=1,
            step=0.1,
            tooltip="Minimum probability threshold for isotope transmission (used with 'probability' mode)",
        )

        number_field(
            label="Max Isotopes to Transmit",
            value=state.config.quad_transmission.quad_transmission_max_isotopes,
            on_change=lambda v: setattr(
                state.config.quad_transmission, "quad_transmission_max_isotopes", int(v)
            ),
            min_val=1,
            max_val=20,
            step=1,
            tooltip="Maximum number of isotope peaks to consider for transmission",
        )


def create_noise_panel(state: "SimulationState") -> None:
    """Create the Noise configuration panel."""
    with ui.column().classes("w-full gap-4 p-4"):
        ui.label("Abundance Noise").classes("text-lg font-bold")

        toggle_field(
            label="Frame Abundance Noise",
            value=state.config.noise.noise_frame_abundance,
            on_change=lambda v: setattr(state.config.noise, "noise_frame_abundance", v),
            tooltip="Add intensity variation between frames",
        )

        toggle_field(
            label="Scan Abundance Noise",
            value=state.config.noise.noise_scan_abundance,
            on_change=lambda v: setattr(state.config.noise, "noise_scan_abundance", v),
            tooltip="Add intensity variation between scans within a frame",
        )

        ui.separator()

        ui.label("M/Z Noise").classes("text-lg font-bold mt-4")

        toggle_field(
            label="Precursor M/Z Noise",
            value=state.config.noise.mz_noise_precursor,
            on_change=lambda v: setattr(state.config.noise, "mz_noise_precursor", v),
            tooltip="Add mass accuracy error to precursor ions",
        )

        number_field(
            label="Precursor Noise",
            value=state.config.noise.precursor_noise_ppm,
            on_change=lambda v: setattr(state.config.noise, "precursor_noise_ppm", v),
            min_val=0,
            step=0.5,
            suffix="ppm",
            tooltip="Mass accuracy error for precursor ions in ppm",
        )

        toggle_field(
            label="Fragment M/Z Noise",
            value=state.config.noise.mz_noise_fragment,
            on_change=lambda v: setattr(state.config.noise, "mz_noise_fragment", v),
            tooltip="Add mass accuracy error to fragment ions",
        )

        number_field(
            label="Fragment Noise",
            value=state.config.noise.fragment_noise_ppm,
            on_change=lambda v: setattr(state.config.noise, "fragment_noise_ppm", v),
            min_val=0,
            step=0.5,
            suffix="ppm",
            tooltip="Mass accuracy error for fragment ions in ppm",
        )

        toggle_field(
            label="Uniform M/Z Noise",
            value=state.config.noise.mz_noise_uniform,
            on_change=lambda v: setattr(state.config.noise, "mz_noise_uniform", v),
            tooltip="Use uniform distribution instead of Gaussian",
        )

        ui.separator()

        ui.label("Real Data Noise").classes("text-lg font-bold mt-4")

        toggle_field(
            label="Add Real Data Noise",
            value=state.config.noise.add_real_data_noise,
            on_change=lambda v: setattr(state.config.noise, "add_real_data_noise", v),
            tooltip="Add chemical/electronic noise from reference data",
        )

        number_field(
            label="Max Reference Noise Intensity",
            value=state.config.noise.reference_noise_intensity_max,
            on_change=lambda v: setattr(state.config.noise, "reference_noise_intensity_max", v),
            min_val=0,
            step=10,
        )

        with ui.row().classes("w-full gap-4"):
            number_field(
                label="Precursor Sample Fraction",
                value=state.config.noise.precursor_sample_fraction,
                on_change=lambda v: setattr(state.config.noise, "precursor_sample_fraction", v),
                min_val=0,
                max_val=1,
                step=0.05,
            )

            number_field(
                label="Fragment Sample Fraction",
                value=state.config.noise.fragment_sample_fraction,
                on_change=lambda v: setattr(state.config.noise, "fragment_sample_fraction", v),
                min_val=0,
                max_val=1,
                step=0.05,
            )

        with ui.row().classes("w-full gap-4"):
            number_field(
                label="Precursor Noise Frames",
                value=state.config.noise.num_precursor_noise_frames,
                on_change=lambda v: setattr(
                    state.config.noise, "num_precursor_noise_frames", int(v)
                ),
                min_val=1,
                step=1,
            )

            number_field(
                label="Fragment Noise Frames",
                value=state.config.noise.num_fragment_noise_frames,
                on_change=lambda v: setattr(
                    state.config.noise, "num_fragment_noise_frames", int(v)
                ),
                min_val=1,
                step=1,
            )


def create_acquisition_panel(state: "SimulationState") -> None:
    """Create the Acquisition configuration panel."""
    with ui.column().classes("w-full gap-4 p-4"):
        ui.label("General Acquisition Settings").classes("text-lg font-bold")

        toggle_field(
            label="Round Collision Energy",
            value=state.config.acquisition.round_collision_energy,
            on_change=lambda v: setattr(state.config.acquisition, "round_collision_energy", v),
        )

        number_field(
            label="Collision Energy Decimals",
            value=state.config.acquisition.collision_energy_decimals,
            on_change=lambda v: setattr(
                state.config.acquisition, "collision_energy_decimals", int(v)
            ),
            min_val=0,
            max_val=3,
            step=1,
        )

        ui.separator()

        ui.label("DDA-PASEF Settings").classes("text-lg font-bold mt-4")
        ui.label("(Only applies when acquisition type is DDA)").classes(
            "text-sm text-gray-500 -mt-2"
        )

        number_field(
            label="Precursor Scan Every N Frames",
            value=state.config.dda.precursors_every,
            on_change=lambda v: setattr(state.config.dda, "precursors_every", int(v)),
            min_val=1,
            step=1,
        )

        number_field(
            label="Precursor Intensity Threshold",
            value=state.config.dda.precursor_intensity_threshold,
            on_change=lambda v: setattr(
                state.config.dda, "precursor_intensity_threshold", int(v)
            ),
            min_val=0,
            step=100,
        )

        number_field(
            label="Max Precursors per Cycle",
            value=state.config.dda.max_precursors,
            on_change=lambda v: setattr(state.config.dda, "max_precursors", int(v)),
            min_val=1,
            step=1,
        )

        number_field(
            label="Exclusion Width (frames)",
            value=state.config.dda.exclusion_width,
            on_change=lambda v: setattr(state.config.dda, "exclusion_width", int(v)),
            min_val=0,
            step=5,
        )

        select_field(
            label="Selection Mode",
            value=state.config.dda.selection_mode,
            options=["topN", "random"],
            on_change=lambda v: setattr(state.config.dda, "selection_mode", v),
        )

        ui.separator()

        ui.label("Proteome Mix Settings").classes("text-lg font-bold mt-4")

        toggle_field(
            label="Enable Proteome Mix Mode",
            value=state.config.proteome_mix.proteome_mix,
            on_change=lambda v: setattr(state.config.proteome_mix, "proteome_mix", v),
            tooltip="Enable multi-FASTA mixing with different dilutions",
        )

        text_field(
            label="Multi-FASTA Dilution (JSON)",
            value=state.config.proteome_mix.multi_fasta_dilution,
            on_change=lambda v: setattr(
                state.config.proteome_mix, "multi_fasta_dilution", v
            ),
            placeholder='{"file1.fasta": 1.0, "file2.fasta": 0.5}',
        )

        ui.separator()

        ui.label("Phosphorylation Settings").classes("text-lg font-bold mt-4")

        toggle_field(
            label="Phosphorylation Mode",
            value=state.config.phosphorylation.phospho_mode,
            on_change=lambda v: setattr(state.config.phosphorylation, "phospho_mode", v),
            tooltip="Enable phosphoproteomics simulation",
        )


def create_3d_preview_panel(state: "SimulationState") -> None:
    """Create the 3D Preview panel for point cloud visualization."""
    from pathlib import Path

    # Track current paths for the visualizations
    ref_current_path = [state.config.paths.reference_path or ""]
    sim_current_path = [""]

    def get_sim_output_path() -> str:
        """Get the expected simulation output .d folder path."""
        save_path = state.config.paths.save_path
        if not save_path:
            return ""
        exp_name = state.config.experiment.experiment_name.replace(
            "[PLACEHOLDER]", state.config.experiment.acquisition_type
        )
        exp_name = exp_name.replace(" ", "_").replace("/", "_")
        return str(Path(save_path) / f"{exp_name}.d")

    sim_current_path[0] = get_sim_output_path()

    with ui.column().classes("w-full gap-4 p-4"):
        # Reference dataset visualization
        ui.label("Reference Dataset").classes("text-xl font-bold text-blue-400")

        # We need to create the visualization first to reference it in the callback
        ref_vis_component = create_3d_visualization(
            initial_data_path=ref_current_path[0],
            max_points=30000,
        )

        # Path picker for reference - inserted before the visualization card
        def on_ref_path_change(path: str):
            ref_current_path[0] = path
            ref_vis_component["set_data_path"](path)

        ref_path_input = path_input(
            label="Reference .d folder",
            value=ref_current_path[0],
            on_change=on_ref_path_change,
            placeholder="/path/to/reference.d",
            directory=True,
        )

        ui.separator().classes("my-4")

        # Simulated dataset visualization
        ui.label("Simulated Dataset").classes("text-xl font-bold text-green-400")

        sim_vis_component = create_3d_visualization(
            initial_data_path=sim_current_path[0],
            max_points=30000,
        )

        # Path picker for simulated
        def on_sim_path_change(path: str):
            sim_current_path[0] = path
            sim_vis_component["set_data_path"](path)

        sim_path_input = path_input(
            label="Simulated .d folder",
            value=sim_current_path[0],
            on_change=on_sim_path_change,
            placeholder="/path/to/simulated.d",
            directory=True,
        )

        # Store references so we can update when paths change
        state._3d_vis_component = ref_vis_component
        state._3d_sim_vis_component = sim_vis_component
        state._3d_ref_path_input = ref_path_input
        state._3d_sim_path_input = sim_path_input

        # Helper to update paths from state
        def update_ref_path():
            new_path = state.config.paths.reference_path or ""
            ref_current_path[0] = new_path
            ref_path_input.value = new_path
            ref_vis_component["set_data_path"](new_path)

        def update_sim_path():
            sim_path = get_sim_output_path()
            sim_current_path[0] = sim_path
            sim_path_input.value = sim_path
            sim_vis_component["set_data_path"](sim_path)

        state._update_ref_3d_path = update_ref_path
        state._update_sim_3d_path = update_sim_path


def create_performance_panel(state: "SimulationState") -> None:
    """Create the Performance configuration panel."""
    with ui.column().classes("w-full gap-4 p-4"):
        ui.label("Threading & Batching").classes("text-lg font-bold")

        number_field(
            label="Number of Threads",
            value=state.config.performance.num_threads,
            on_change=lambda v: setattr(state.config.performance, "num_threads", int(v)),
            min_val=-1,
            step=1,
            tooltip="-1 for auto-detect",
        )

        number_field(
            label="Batch Size",
            value=state.config.performance.batch_size,
            on_change=lambda v: setattr(state.config.performance, "batch_size", int(v)),
            min_val=1,
            step=64,
            tooltip="Number of peptides to process per batch",
        )

        number_field(
            label="Frame Batch Size",
            value=state.config.performance.frame_batch_size,
            on_change=lambda v: setattr(state.config.performance, "frame_batch_size", int(v)),
            min_val=1,
            step=100,
            tooltip="Number of frames to write per batch",
        )

        toggle_field(
            label="Lazy Frame Assembly",
            value=state.config.performance.lazy_frame_assembly,
            on_change=lambda v: setattr(state.config.performance, "lazy_frame_assembly", v),
            tooltip="Reduces memory usage at slight performance cost",
        )

        ui.separator()

        ui.label("GPU Settings").classes("text-lg font-bold mt-4")

        toggle_field(
            label="Enable GPU Acceleration",
            value=state.config.performance.use_gpu,
            on_change=lambda v: setattr(state.config.performance, "use_gpu", v),
            tooltip="Use GPU for deep learning predictions",
        )

        number_field(
            label="GPU Memory Limit",
            value=state.config.performance.gpu_memory_limit_gb,
            on_change=lambda v: setattr(state.config.performance, "gpu_memory_limit_gb", int(v)),
            min_val=1,
            max_val=32,
            step=1,
            suffix="GB",
            tooltip="Maximum GPU memory to allocate for deep learning",
        )

        ui.separator()

        ui.label("Logging").classes("text-lg font-bold mt-4")

        select_field(
            label="Log Level",
            value=state.config.logging.log_level,
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            on_change=lambda v: setattr(state.config.logging, "log_level", v),
        )

        text_field(
            label="Log File (optional)",
            value=state.config.logging.log_file,
            on_change=lambda v: setattr(state.config.logging, "log_file", v),
            placeholder="Leave empty for console only",
        )


async def load_config_dialog(state: "SimulationState") -> None:
    """Show dialog to load configuration file."""
    from pathlib import Path
    from ..state import find_config_in_path, get_recent_configs

    recent_configs = get_recent_configs()

    with ui.dialog() as dialog, ui.card().classes("w-[500px]"):
        ui.label("Load Configuration").classes("text-lg font-bold mb-4")

        # Manual path input
        input_field = ui.input(
            label="Configuration File or Simulation Directory",
            placeholder="/path/to/config.toml or /path/to/simulation.d",
        ).classes("w-full")

        async def browse():
            browser = FileBrowser(
                start_path=input_field.value or str(Path.home()),
                select_directory=True,
                file_filter=[".toml"],
                title="Select Configuration File or Simulation Directory",
                show_files_with_directory=True,  # Show .toml files AND directories
            )
            path = await browser.pick()
            if path:
                input_field.value = path

        with ui.row().classes("w-full justify-end"):
            ui.button("Browse...", icon="folder_open", on_click=browse).props("flat")

        # Recent configs
        if recent_configs:
            ui.separator().classes("my-4")
            ui.label("Recent Configurations").classes("font-medium text-gray-700 mb-2")

            for config in recent_configs[:5]:
                config_path = config.get("path", "")
                config_name = config.get("name", Path(config_path).stem)

                def select_recent(p=config_path):
                    input_field.value = p

                with ui.row().classes("w-full items-center hover:bg-gray-100 p-2 rounded cursor-pointer").on(
                    "click", select_recent
                ):
                    ui.icon("history", color="gray").classes("text-lg")
                    with ui.column().classes("flex-grow ml-2"):
                        ui.label(config_name).classes("font-medium text-sm")
                        ui.label(config_path).classes("text-xs text-gray-500 truncate")

        def load_file():
            path = input_field.value
            if not path:
                ui.notify("Please enter a path", type="warning")
                return

            config_path = find_config_in_path(path)
            if not config_path:
                ui.notify(f"No config file found in: {path}", type="negative")
                return

            success, message = state.load_config(config_path)
            if success:
                ui.notify(message, type="positive")
                dialog.close()
                ui.navigate.reload()
            else:
                ui.notify(message, type="negative")

        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Load", on_click=load_file).props("color=primary")

    dialog.open()


async def save_config_dialog(state: "SimulationState") -> None:
    """Show dialog to save configuration file."""
    from pathlib import Path

    # Generate default filename
    default_name = state.config.experiment.experiment_name.replace(
        "[PLACEHOLDER]", state.config.experiment.acquisition_type
    )
    default_name = default_name.replace(" ", "_").replace("/", "_")

    with ui.dialog() as dialog, ui.card().classes("w-[500px]"):
        ui.label("Save Configuration").classes("text-lg font-bold mb-4")

        # Directory input
        dir_input = ui.input(
            label="Save Directory",
            value=state.loaded_config_path and str(Path(state.loaded_config_path).parent) or "",
            placeholder="/path/to/directory",
        ).classes("w-full")

        async def browse():
            browser = FileBrowser(
                start_path=dir_input.value or str(Path.home()),
                select_directory=True,
                title="Select Directory to Save Configuration",
            )
            path = await browser.pick()
            if path:
                dir_input.value = path

        with ui.row().classes("w-full justify-end"):
            ui.button("Browse...", icon="folder_open", on_click=browse).props("flat")

        # Filename input
        name_input = ui.input(
            label="Filename",
            value=f"{default_name}_config.toml",
            placeholder="config.toml",
        ).classes("w-full mt-2")

        def save_file():
            directory = dir_input.value
            filename = name_input.value

            if not directory:
                ui.notify("Please select a directory", type="warning")
                return

            if not filename:
                ui.notify("Please enter a filename", type="warning")
                return

            # Ensure .toml extension
            if not filename.endswith(".toml"):
                filename += ".toml"

            full_path = str(Path(directory) / filename)
            success, message = state.save_config(full_path)
            if success:
                ui.notify(message, type="positive")
                dialog.close()
            else:
                ui.notify(message, type="negative")

        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Save", on_click=save_file).props("color=primary")

    dialog.open()


def create_help_panel() -> None:
    """Create the Help/User Manual panel with documentation for all settings."""
    with ui.column().classes("w-full gap-4 p-4"):
        ui.label("TimSim User Manual").classes("text-2xl font-bold")
        ui.label(
            "This manual describes all configuration options for TimSim. "
            "Hover over any field in the configuration tabs for quick tooltips."
        ).classes("text-gray-400 mb-4")

        # Use expansion panels for each section
        with ui.expansion("Files & Paths", icon="folder").classes("w-full"):
            ui.markdown("""
**Reference Dataset (.d folder)**
Path to a Bruker timsTOF .d folder used as a template for the simulation.
The reference provides acquisition parameters, frame structure, and optionally noise patterns.

**FASTA File**
Protein sequence database in FASTA format. Proteins will be digested in-silico
to generate peptides for simulation.

**Output Directory**
Where the simulated .d dataset will be saved. The experiment name determines the folder name.

**Experiment Name**
Name for the simulation run. Use [PLACEHOLDER] to auto-insert the acquisition type (DIA/DDA).

**Acquisition Type**
- **DIA** (Data-Independent Acquisition): All precursors in isolation windows are fragmented
- **DDA** (Data-Dependent Acquisition): Top-N most intense precursors are selected for fragmentation

**Gradient Length**
Total LC gradient duration in seconds. Affects retention time distribution of peptides.
            """)

        with ui.expansion("Digestion Settings", icon="content_cut").classes("w-full"):
            ui.markdown("""
**Number of Proteins**
Maximum number of proteins to load from the FASTA file.

**Total Peptides**
Total number of peptides to generate from protein digestion.

**Sample Peptides**
Number of peptides to randomly sample for simulation (reduces computation time).

**Missed Cleavages**
Number of allowed missed enzymatic cleavage sites (0-3 typical).

**Min/Max Peptide Length**
Filter peptides by length (amino acids). Typical range: 7-30.

**Min/Max Peptide Mass**
Filter peptides by mass (Da). Typical range: 400-6000.

**Cleave At / Skip At**
Enzyme specificity rules. For trypsin: cleave at K,R and skip at P (proline rule).
            """)

        with ui.expansion("Chromatography (Retention Time)", icon="science").classes("w-full"):
            ui.markdown("""
**Retention Time Model**
Prediction model for peptide retention times. Deep learning models provide better accuracy.

**EMG Peak Shape Parameters**
Peaks are modeled using Exponentially Modified Gaussian (EMG) distribution:
- **Mean (μ)**: Center of the Gaussian component
- **Std (σ)**: Width of the Gaussian component
- **Lambda (λ)**: Exponential decay rate (controls tailing)
- **Min/Max Std**: Range for peak width variation

**Sampling Rate**
Points per second for peak shape sampling. Higher = smoother peaks but more computation.
            """)

        with ui.expansion("Ion Settings", icon="show_chart").classes("w-full"):
            ui.markdown("""
### Charge States
- **Charge Probability (p)**: Probability parameter for charge state distribution
- **Maximum Charge**: Highest charge state to consider (typically 4-5)
- **Min Charge Contribution**: Minimum abundance threshold for a charge state
- **Binomial Charge Model**: Use simpler binomial model instead of ML prediction
- **Charge +1 Probability**: Explicit probability for singly charged ions

### Isotopic Pattern
- **Number of Isotope Peaks**: How many isotope peaks to simulate (typically 5-10)
- **Min Isotope Intensity**: Threshold for including isotope peaks
- **Centroid Isotopes**: Output centroided (single m/z) rather than profile peaks

### Prediction Models (Koina Integration)
TimSim supports both local PyTorch models and online Koina API models:

**RT (Retention Time) Models:**
- Local PyTorch: Fast, runs locally
- Deeplc_hela_hf: DeepLC model via Koina
- Chronologer_RT: Chronologer model via Koina
- AlphaPeptDeep_rt_generic: AlphaPeptDeep model via Koina (supports phospho)

**CCS (Ion Mobility) Models:**
- Local PyTorch: Fast, runs locally
- AlphaPeptDeep_ccs_generic: AlphaPeptDeep model via Koina (supports phospho)
- IM2Deep: IM2Deep model via Koina

**Fragment Intensity Models:**
- Prosit: Prosit_2023_intensity_timsTOF via Koina (default)
- AlphaPeptDeep: AlphaPeptDeep_ms2_generic via Koina (supports phospho!)
- MS2PIP: ms2pip_timsTOF2024 via Koina

**Important:** For phosphorylated peptides, use AlphaPeptDeep models as Prosit doesn't support phospho modifications.

### Fragment Intensity
- **Downsampling Factor**: Reduce fragment peaks for performance (0.5 = 50% kept)

### Quadrupole Transmission
Controls how isotope patterns are affected by quadrupole isolation:
- **none**: All isotopes transmitted equally
- **precursor_scaling**: Scale isotopes based on precursor transmission
- **per_fragment**: Accurate recalculation for each fragment ion

- **Min Transmission Probability**: Threshold for isotope transmission
- **Max Isotopes**: Maximum isotope peaks to consider
            """)

        with ui.expansion("Noise Settings", icon="grain").classes("w-full"):
            ui.markdown("""
### Abundance Noise
- **Frame Abundance Noise**: Add intensity variation between frames
- **Scan Abundance Noise**: Add intensity variation between scans

### M/z Noise
- **Precursor m/z Noise**: Add mass accuracy error to precursor ions (ppm)
- **Fragment m/z Noise**: Add mass accuracy error to fragment ions (ppm)
- **Uniform m/z Noise**: Use uniform instead of Gaussian distribution

### Real Data Noise
Inject noise patterns from the reference dataset:
- **Add Real Data Noise**: Enable noise injection from reference
- **Noise Intensity Max**: Maximum intensity for sampled noise peaks
- **Precursor/Fragment Sample Fraction**: Fraction of noise peaks to sample
- **Noise Frames**: Number of reference frames to sample noise from
            """)

        with ui.expansion("Acquisition Settings", icon="settings").classes("w-full"):
            ui.markdown("""
### DDA Settings (Data-Dependent Acquisition)
- **TopN**: Number of precursors to select per cycle
- **Min Intensity**: Minimum precursor intensity for selection
- **Exclusion Time**: Dynamic exclusion duration (seconds)

### Acquisition Mode
Select the PASEF acquisition scheme:
- **Standard DIA/DDA**: Basic acquisition without special PASEF patterns
- **Synchro-PASEF**: Synchronized parallel accumulation
- **midia-PASEF**: Multiplexed DIA with interleaved windows
- **Slice-PASEF**: Slice-based acquisition for increased coverage

### Visualization
The acquisition mode plot shows the isolation window pattern across the m/z and mobility dimensions.
            """)

        with ui.expansion("Performance Settings", icon="speed").classes("w-full"):
            ui.markdown("""
**Number of Threads**
Parallel processing threads. Set to -1 for auto-detection (uses all CPU cores).

**Batch Size**
Number of peptides to process in each batch. Larger = more memory, potentially faster.

**Frame Batch Size**
Number of frames to write in each batch.

**Lazy Frame Assembly**
Build frames on-demand instead of pre-computing. Reduces memory usage for large simulations.
            """)

        with ui.expansion("3D Preview", icon="view_in_ar").classes("w-full"):
            ui.markdown("""
**3D Point Cloud Visualization**
Interactive visualization of m/z vs Ion Mobility vs Retention Time.

- **Frame Type**: Choose between Precursor (MS1) and Fragment (MS2) frames
- **Max Points**: Limit total points for smooth interaction (30,000 default)
- **Frames to Sample**: Number of frames to load (evenly distributed across run)

Use the visualization to compare reference data with simulated output and verify
the simulation quality.
            """)

        with ui.expansion("Workflow Tips", icon="lightbulb").classes("w-full"):
            ui.markdown("""
### Recommended Workflow

1. **Set Paths**: Configure reference dataset, FASTA, and output directory
2. **Choose Acquisition**: Select DIA or DDA and the PASEF mode
3. **Adjust Peptide Count**: Start with fewer peptides for testing
4. **Configure Ion Settings**: Adjust charge states and isotope patterns
5. **Add Noise (Optional)**: Enable noise for realistic output
6. **Run Simulation**: Monitor progress in the console
7. **Verify Output**: Use 3D Preview to compare reference vs simulated data

### Performance Tips
- Use **Lazy Frame Assembly** for large simulations (>100k peptides)
- Reduce **Max Points** in 3D Preview if visualization is slow
- Enable **GPU** if available (requires CUDA-capable GPU)
- Adjust **Batch Size** based on available RAM

### Troubleshooting
- **No output**: Check that output directory exists and is writable
- **Memory errors**: Enable lazy frame assembly, reduce batch size
- **Slow simulation**: Reduce peptide count, enable threading
- **Missing peaks**: Check charge state and isotope settings
            """)

        ui.separator()
        ui.label("For more information, see the TimSim documentation.").classes("text-sm text-gray-500")
