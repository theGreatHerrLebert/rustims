"""Log console component for TimSim GUI."""

from typing import Optional
from nicegui import ui


class LogConsole:
    """A scrollable log console for displaying simulation output."""

    def __init__(self, max_lines: int = 1000):
        """Create a log console.

        Args:
            max_lines: Maximum number of lines to keep in buffer
        """
        self.max_lines = max_lines
        self._lines: list[str] = []
        self._container: Optional[ui.card] = None
        self._log_element: Optional[ui.element] = None

    def create(self) -> "LogConsole":
        """Create the console UI elements."""
        with ui.card().classes("w-full") as self._container:
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label("Console Output").classes("font-bold text-lg")
                with ui.row().classes("gap-2"):
                    ui.button(
                        icon="delete",
                        on_click=self.clear,
                    ).props("flat dense").tooltip("Clear console")
                    ui.button(
                        icon="content_copy",
                        on_click=self._copy_to_clipboard,
                    ).props("flat dense").tooltip("Copy to clipboard")

            # Use a simple label inside a scroll area with monospace styling
            with ui.scroll_area().classes("w-full h-64 border rounded bg-gray-900") as self._scroll:
                self._log_element = ui.label("").classes("text-green-400 font-mono text-xs whitespace-pre-wrap p-2")

        return self

    def push(self, message: str) -> None:
        """Add a message to the console.

        Args:
            message: Message to add
        """
        self._lines.append(message)
        if len(self._lines) > self.max_lines:
            self._lines = self._lines[-self.max_lines :]

        if self._log_element:
            try:
                self._log_element.text = "\n".join(self._lines)
                # Auto-scroll to bottom
                if hasattr(self, '_scroll') and self._scroll:
                    self._scroll.scroll_to(percent=1.0)
            except Exception as e:
                print(f"[CONSOLE] push failed: {e}", flush=True)

    def clear(self) -> None:
        """Clear all console output."""
        self._lines = []
        if self._log_element:
            self._log_element.text = ""

    def _copy_to_clipboard(self) -> None:
        """Copy console contents to clipboard."""
        content = "\n".join(self._lines)
        ui.run_javascript(f"navigator.clipboard.writeText({repr(content)})")
        ui.notify("Copied to clipboard", type="positive")

    def get_lines(self) -> list[str]:
        """Get all console lines."""
        return self._lines.copy()


def create_progress_section() -> tuple[ui.linear_progress, ui.label]:
    """Create a progress bar with status label.

    Returns:
        Tuple of (progress_bar, status_label)
    """
    with ui.column().classes("w-full gap-2"):
        status_label = ui.label("Ready").classes("text-gray-600")
        progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")

    return progress_bar, status_label
