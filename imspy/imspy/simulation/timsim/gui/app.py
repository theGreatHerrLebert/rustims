"""Main NiceGUI application for TimSim."""

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Optional

from nicegui import ui, app, context, background_tasks

from .state import SimulationState, get_recent_configs, find_config_in_path
from .pages.advanced import create_advanced_view
from .components.file_inputs import FileBrowser


# Global state instance
state = SimulationState()

# Process reference for running simulation
_simulation_process: Optional[asyncio.subprocess.Process] = None


async def show_startup_dialog():
    """Show startup dialog with options to start fresh or load config."""
    recent_configs = get_recent_configs()

    result = {"action": "new", "path": None}

    with ui.dialog() as dialog, ui.card().classes("w-[500px]"):
        ui.label("Welcome to TimSim").classes("text-2xl font-bold mb-4")
        ui.label("Proteomics Simulation Engine for timsTOF").classes("text-gray-600 mb-6")

        # Options
        def submit_new():
            result.update({"action": "new"})
            dialog.submit(result)

        with ui.column().classes("w-full gap-4"):
            # New simulation
            with ui.card().classes("w-full cursor-pointer hover:shadow-lg").on("click", submit_new):
                with ui.row().classes("items-center gap-4"):
                    ui.icon("add_circle", color="primary").classes("text-3xl")
                    with ui.column():
                        ui.label("Start New Simulation").classes("font-bold")
                        ui.label("Create a new simulation from scratch").classes("text-sm text-gray-600")

            # Load config
            async def browse_config():
                browser = FileBrowser(
                    start_path=str(Path.home()),
                    select_directory=True,
                    file_filter=[".toml"],
                    title="Select Configuration File or Simulation Directory",
                    show_files_with_directory=True,  # Show .toml files AND directories
                )
                path = await browser.pick()
                if path:
                    config_path = find_config_in_path(path)
                    if config_path:
                        result.update({"action": "load", "path": config_path})
                        dialog.submit(result)
                    else:
                        ui.notify("No config file found in selected location", type="warning")

            with ui.card().classes("w-full cursor-pointer hover:shadow-lg").on("click", browse_config):
                with ui.row().classes("items-center gap-4"):
                    ui.icon("folder_open", color="amber").classes("text-3xl")
                    with ui.column():
                        ui.label("Load Configuration").classes("font-bold")
                        ui.label("Load from TOML file or simulation folder").classes("text-sm text-gray-600")

            # Recent configs
            if recent_configs:
                ui.separator().classes("my-2")
                ui.label("Recent Configurations").classes("font-medium text-gray-700")

                for config in recent_configs[:5]:
                    config_path = config.get("path", "")
                    config_name = config.get("name", Path(config_path).stem)

                    def load_recent(p=config_path):
                        result.update({"action": "load", "path": p})
                        dialog.submit(result)

                    with ui.row().classes("w-full items-center hover:bg-gray-100 p-2 rounded cursor-pointer").on(
                        "click", load_recent
                    ):
                        ui.icon("history", color="gray").classes("text-lg")
                        with ui.column().classes("flex-grow ml-2"):
                            ui.label(config_name).classes("font-medium")
                            ui.label(config_path).classes("text-xs text-gray-500 truncate")

    dialog.open()
    return await dialog


def safe_client_op(client, func):
    """Safely execute a function within client context, handling deleted clients."""
    if client is None:
        return
    try:
        with client:
            func()
    except Exception:
        pass  # Client was deleted, silently ignore


async def run_simulation(client=None):
    """Run the TimSim simulation."""
    global _simulation_process

    # Use passed client or try to get from context
    if client is None:
        try:
            client = context.client
        except RuntimeError:
            return

    # Validate configuration
    valid, errors, warnings = state.validate()

    # Show warnings (with client context)
    try:
        with client:
            for warning in warnings:
                ui.notify(warning, type="warning", timeout=5000)

            if not valid:
                for error in errors:
                    ui.notify(error, type="negative", timeout=5000)
                return

            if state.is_running:
                ui.notify("Simulation already running", type="warning")
                return
    except Exception:
        if not valid:
            return

    state.is_running = True

    # Update UI (with client context)
    try:
        with client:
            if hasattr(state, "_status_label"):
                state._status_label.text = "Running simulation..."
            if hasattr(state, "_progress_bar"):
                state._progress_bar.value = 0
            if hasattr(state, "_toggle_run_button"):
                state._toggle_run_button()

            # Clear console
            if hasattr(state, "_console"):
                state._console.clear()
                state._console.push("Starting TimSim simulation...")
    except Exception:
        pass

    # Save config to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as config_file:
        config_path = config_file.name
        state.config.save_toml(config_path)

    try:
        with client:
            if hasattr(state, "_console"):
                state._console.push(f"Config saved to: {config_path}")
    except Exception:
        pass

    try:
        # Run timsim as subprocess using asyncio for non-blocking I/O
        # Use -u flag for unbuffered output to ensure real-time streaming
        cmd = [sys.executable, "-u", "-m", "imspy.simulation.timsim.simulator", config_path]
        print(f"[TIMSIM] Starting subprocess: {' '.join(cmd)}", flush=True)

        _simulation_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        print(f"[TIMSIM] Process started with PID: {_simulation_process.pid}", flush=True)

        # Stream output to console (non-blocking)
        all_output = []
        while True:
            if _simulation_process.stdout is None:
                break

            line = await _simulation_process.stdout.readline()
            if not line:
                break

            line_text = line.decode().rstrip()
            if line_text:
                all_output.append(line_text)
                print(f"[TIMSIM] {line_text}", flush=True)  # Debug: log to server console
                try:
                    with client:
                        if hasattr(state, "_console"):
                            state._console.push(line_text)

                        # Parse progress from log lines
                        if "frame assembly" in line_text.lower():
                            try:
                                # Try to extract progress percentage
                                if "%" in line_text:
                                    pct = float(line_text.split("%")[0].split()[-1])
                                    if hasattr(state, "_progress_bar"):
                                        state._progress_bar.value = pct / 100
                            except (ValueError, IndexError):
                                pass
                except Exception:
                    pass  # Client deleted, continue simulation anyway

        print(f"[TIMSIM] Output loop finished, waiting for process...", flush=True)
        await _simulation_process.wait()
        return_code = _simulation_process.returncode
        print(f"[TIMSIM] Process finished with return code: {return_code}", flush=True)
        if all_output:
            print(f"[TIMSIM] Total output lines: {len(all_output)}", flush=True)
        else:
            print("[TIMSIM] WARNING: No output captured from subprocess!", flush=True)

        try:
            with client:
                if return_code == 0:
                    if hasattr(state, "_console"):
                        state._console.push("Simulation completed successfully!")
                    if hasattr(state, "_status_label"):
                        state._status_label.text = "Completed"
                    if hasattr(state, "_progress_bar"):
                        state._progress_bar.value = 1.0
                    ui.notify("Simulation completed!", type="positive")
                else:
                    if hasattr(state, "_console"):
                        state._console.push(f"Simulation failed with return code: {return_code}")
                    if hasattr(state, "_status_label"):
                        state._status_label.text = "Failed"
                    ui.notify(f"Simulation failed (code {return_code})", type="negative")
        except Exception:
            pass  # Client deleted

    except Exception as e:
        try:
            with client:
                if hasattr(state, "_console"):
                    state._console.push(f"Error: {str(e)}")
                if hasattr(state, "_status_label"):
                    state._status_label.text = "Error"
                ui.notify(f"Error: {str(e)}", type="negative")
        except Exception:
            pass  # Client deleted

    finally:
        state.is_running = False
        _simulation_process = None

        # Update button state
        try:
            with client:
                if hasattr(state, "_toggle_run_button"):
                    state._toggle_run_button()
        except Exception:
            pass  # Client deleted

        # Clean up temp config file
        try:
            Path(config_path).unlink()
        except Exception:
            pass


async def stop_simulation(client=None):
    """Stop the running simulation."""
    global _simulation_process

    # Use passed client or try to get from context
    if client is None:
        try:
            client = context.client
        except RuntimeError:
            pass  # Continue without UI updates if no context

    if _simulation_process is not None:
        _simulation_process.terminate()
        try:
            await asyncio.wait_for(_simulation_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            _simulation_process.kill()
        _simulation_process = None
        state.is_running = False

        if client:
            with client:
                if hasattr(state, "_status_label"):
                    state._status_label.text = "Stopped"
                if hasattr(state, "_console"):
                    state._console.push("Simulation stopped by user.")
                if hasattr(state, "_toggle_run_button"):
                    state._toggle_run_button()
                ui.notify("Simulation stopped", type="warning")


async def load_config_with_browser():
    """Open file browser to load a config."""
    browser = FileBrowser(
        start_path=state.loaded_config_path or str(Path.home()),
        select_directory=True,  # Allow selecting .d folders too
        file_filter=[".toml"],
        title="Select Configuration File or Simulation Directory",
    )
    path = await browser.pick()
    if path:
        success, message = state.load_config(path)
        if success:
            ui.notify(message, type="positive")
            # Refresh the page to show new values
            ui.navigate.reload()
        else:
            ui.notify(message, type="negative")


async def save_config_with_browser():
    """Open file browser to save a config."""
    browser = FileBrowser(
        start_path=state.loaded_config_path or str(Path.home()),
        select_directory=True,
        title="Select Directory to Save Configuration",
    )
    path = await browser.pick()
    if path:
        # If directory, add filename
        p = Path(path)
        if p.is_dir():
            # Generate filename from experiment name
            name = state.config.experiment.experiment_name.replace("[PLACEHOLDER]", state.config.experiment.acquisition_type)
            name = name.replace(" ", "_").replace("/", "_")
            path = str(p / f"{name}_config.toml")

        success, message = state.save_config(path)
        if success:
            ui.notify(message, type="positive")
        else:
            ui.notify(message, type="negative")


@ui.page("/")
def main_page():
    """Main page - advanced configuration view."""

    with ui.column().classes("w-full min-h-screen"):
        # Capture client at page render time for use in callbacks
        def start_simulation():
            client = context.client
            background_tasks.create(run_simulation(client))

        def stop_simulation_handler():
            client = context.client
            background_tasks.create(stop_simulation(client))

        create_advanced_view(
            state=state,
            on_run=start_simulation,
            on_stop=stop_simulation_handler,
        )

    # Show startup dialog after page renders (non-blocking)
    async def show_startup():
        if not state.loaded_config_path and state.config.paths.save_path == "":
            result = await show_startup_dialog()
            if result and result.get("action") == "load" and result.get("path"):
                success, message = state.load_config(result["path"])
                if success:
                    ui.notify(message, type="positive")
                    ui.navigate.reload()
                else:
                    ui.notify(message, type="negative")

    ui.timer(0.1, show_startup, once=True)


def main():
    """Entry point for the TimSim GUI."""
    # Configure app
    app.native.window_args["resizable"] = True

    # Run the app
    ui.run(
        title="TimSim - Proteomics Simulation",
        port=8080,
        reload=False,
        show=True,
        favicon="https://raw.githubusercontent.com/theGreatHerrLebert/rustims/main/docs/imspy.svg",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
