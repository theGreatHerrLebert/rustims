"""Simple wizard-style interface for TimSim GUI."""

from typing import Callable, TYPE_CHECKING
from nicegui import ui

from ..components.file_inputs import fasta_picker, reference_picker, output_picker
from ..components.console import LogConsole, create_progress_section

if TYPE_CHECKING:
    from ..state import SimulationState


def create_simple_view(
    state: "SimulationState",
    on_run: Callable[[], None],
    on_stop: Callable[[], None],
    on_switch_mode: Callable[[], None],
) -> ui.element:
    """Create the simple wizard-style interface.

    Args:
        state: Shared simulation state
        on_run: Callback to start simulation
        on_stop: Callback to stop simulation
        on_switch_mode: Callback to switch to advanced mode

    Returns:
        The container element
    """
    wizard_steps = ["Files", "Experiment", "Sample", "Run"]

    # Store UI elements for updates
    step_panels = []
    step_circles = []
    step_labels = []

    def update_step_display():
        """Update visibility and styling based on current step."""
        for i, (panel, circle, label) in enumerate(zip(step_panels, step_circles, step_labels)):
            # Update panel visibility
            panel.visible = (i == state.current_step)

            # Update step circle styling
            if i <= state.current_step:
                circle.classes(replace="w-10 h-10 rounded-full flex items-center justify-center text-white bg-blue-600")
            else:
                circle.classes(replace="w-10 h-10 rounded-full flex items-center justify-center text-white bg-gray-300")

            # Update step label styling
            if i == state.current_step:
                label.classes(replace="mt-2 text-sm font-bold")
            else:
                label.classes(replace="mt-2 text-sm")

        # Update navigation buttons
        back_btn.set_visibility(state.current_step > 0)
        next_btn.set_visibility(state.current_step < len(wizard_steps) - 1)

    def go_back():
        if state.current_step > 0:
            state.current_step -= 1
            update_step_display()

    def go_next():
        if state.current_step < len(wizard_steps) - 1:
            state.current_step += 1
            update_step_display()

    with ui.column().classes("w-full max-w-3xl mx-auto") as container:
        # Header
        with ui.row().classes("w-full items-center justify-between mb-6"):
            ui.label("TimSim - Quick Setup").classes("text-2xl font-bold")
            ui.button(
                "Advanced Mode",
                icon="tune",
                on_click=on_switch_mode,
            ).props("flat")

        # Step indicator
        with ui.row().classes("w-full justify-center mb-8"):
            for i, step in enumerate(wizard_steps):
                with ui.column().classes("items-center mx-4"):
                    circle = ui.element("div").classes(
                        f"w-10 h-10 rounded-full flex items-center justify-center text-white "
                        f"{'bg-blue-600' if i <= state.current_step else 'bg-gray-300'}"
                    )
                    with circle:
                        ui.label(str(i + 1))
                    label = ui.label(step).classes(
                        f"mt-2 text-sm {'font-bold' if i == state.current_step else ''}"
                    )
                    step_circles.append(circle)
                    step_labels.append(label)

        # Step content
        with ui.card().classes("w-full p-6"):
            # Step 0: Files
            step_0 = ui.column().classes("w-full gap-4")
            step_0.visible = (state.current_step == 0)
            with step_0:
                ui.label("Step 1: Select Files").classes("text-xl font-bold mb-4")

                reference_picker(
                    value=state.config.paths.reference_path,
                    on_change=lambda v: setattr(state.config.paths, "reference_path", v),
                )

                fasta_picker(
                    value=state.config.paths.fasta_path,
                    on_change=lambda v: setattr(state.config.paths, "fasta_path", v),
                )

                output_picker(
                    value=state.config.paths.save_path,
                    on_change=lambda v: setattr(state.config.paths, "save_path", v),
                )
            step_panels.append(step_0)

            # Step 1: Experiment Type
            step_1 = ui.column().classes("w-full gap-4")
            step_1.visible = (state.current_step == 1)
            with step_1:
                ui.label("Step 2: Experiment Type").classes("text-xl font-bold mb-4")

                ui.label("Acquisition Mode").classes("font-medium")

                with ui.row().classes("w-full gap-4"):
                    dia_btn = ui.button(
                        "DIA-PASEF",
                        icon="grid_view",
                    ).classes("flex-1 h-20")

                    dda_btn = ui.button(
                        "DDA-PASEF",
                        icon="scatter_plot",
                    ).classes("flex-1 h-20")

                    def update_acq_buttons():
                        is_dia = state.config.experiment.acquisition_type == "DIA"
                        if is_dia:
                            dia_btn.props("color=primary")
                            dda_btn.props(remove="color=primary").props("outline")
                        else:
                            dda_btn.props("color=primary")
                            dia_btn.props(remove="color=primary").props("outline")

                    def set_dia():
                        state.config.experiment.acquisition_type = "DIA"
                        update_acq_buttons()

                    def set_dda():
                        state.config.experiment.acquisition_type = "DDA"
                        update_acq_buttons()

                    dia_btn.on_click(set_dia)
                    dda_btn.on_click(set_dda)
                    update_acq_buttons()

                ui.separator().classes("my-4")

                ui.label("Preset Configuration").classes("font-medium mt-4")
                with ui.row().classes("w-full gap-4"):

                    def apply_preset(preset: str):
                        state.apply_preset(preset)
                        ui.notify(f"Applied {preset} preset", type="info")

                    with ui.column().classes("flex-1"):
                        quick_card = ui.card().classes("cursor-pointer hover:shadow-lg p-4")
                        with quick_card:
                            ui.label("Quick").classes("font-bold")
                            ui.label("5K peptides").classes("text-sm text-gray-600")
                            ui.label("30 min gradient").classes("text-sm text-gray-600")
                        quick_card.on("click", lambda: apply_preset("quick"))

                    with ui.column().classes("flex-1"):
                        std_card = ui.card().classes("cursor-pointer hover:shadow-lg p-4")
                        with std_card:
                            ui.label("Standard").classes("font-bold")
                            ui.label("25K peptides").classes("text-sm text-gray-600")
                            ui.label("60 min gradient").classes("text-sm text-gray-600")
                        std_card.on("click", lambda: apply_preset("standard"))

                    with ui.column().classes("flex-1"):
                        hf_card = ui.card().classes("cursor-pointer hover:shadow-lg p-4")
                        with hf_card:
                            ui.label("High Fidelity").classes("font-bold")
                            ui.label("50K peptides").classes("text-sm text-gray-600")
                            ui.label("120 min gradient").classes("text-sm text-gray-600")
                        hf_card.on("click", lambda: apply_preset("high_fidelity"))

            step_panels.append(step_1)

            # Step 2: Sample Settings
            step_2 = ui.column().classes("w-full gap-4")
            step_2.visible = (state.current_step == 2)
            with step_2:
                ui.label("Step 3: Sample Settings").classes("text-xl font-bold mb-4")

                ui.label("Number of Peptides").classes("font-medium")
                with ui.row().classes("w-full items-center gap-4"):
                    peptide_label = ui.label(
                        f"{state.config.digestion.num_sample_peptides:,}"
                    ).classes("min-w-24 text-right font-mono text-lg")

                    def on_peptide_slider(e):
                        state.config.digestion.num_sample_peptides = int(e.value)
                        peptide_label.text = f"{int(e.value):,}"

                    ui.slider(
                        value=state.config.digestion.num_sample_peptides,
                        min=1000,
                        max=100000,
                        step=1000,
                        on_change=on_peptide_slider,
                    ).classes("flex-grow")

                ui.separator().classes("my-4")

                ui.label("Experiment Name").classes("font-medium")
                ui.input(
                    value=state.config.experiment.experiment_name,
                    on_change=lambda e: setattr(
                        state.config.experiment, "experiment_name", e.value
                    ),
                ).classes("w-full")

                ui.separator().classes("my-4")

                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Add Real Data Noise").classes("font-medium")
                    ui.switch(
                        value=state.config.noise.add_real_data_noise,
                        on_change=lambda e: setattr(
                            state.config.noise, "add_real_data_noise", e.value
                        ),
                    )

            step_panels.append(step_2)

            # Step 3: Run
            step_3 = ui.column().classes("w-full gap-4")
            step_3.visible = (state.current_step == 3)
            with step_3:
                ui.label("Step 4: Run Simulation").classes("text-xl font-bold mb-4")

                # Summary - these labels will be static at creation time
                # For a dynamic summary, we'd need to update on step change
                with ui.card().classes("w-full bg-gray-50 p-4"):
                    ui.label("Configuration Summary").classes("font-bold mb-4")

                    summary_grid = ui.column().classes("w-full gap-2")

                # Progress section
                progress_bar, status_label = create_progress_section()

                # Console
                console = LogConsole()
                console.create()

                # Store references for external access
                state._console = console
                state._progress_bar = progress_bar
                state._status_label = status_label

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
                ).classes("w-full h-16 text-xl")
                run_btn.props("color=positive")

                # Store toggle function for external updates
                state._toggle_run_button = toggle_run_button

            step_panels.append(step_3)

        # Navigation buttons
        with ui.row().classes("w-full justify-between mt-6"):
            back_btn = ui.button("Back", icon="arrow_back", on_click=go_back)
            back_btn.set_visibility(state.current_step > 0)

            ui.space()

            next_btn = ui.button("Next", icon="arrow_forward", on_click=go_next)
            next_btn.props("color=primary")
            next_btn.set_visibility(state.current_step < len(wizard_steps) - 1)

    return container
