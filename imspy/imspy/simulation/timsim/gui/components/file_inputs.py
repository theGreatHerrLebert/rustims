"""File input components for TimSim GUI."""

import os
from pathlib import Path
from typing import Callable, Optional, List
from nicegui import ui


class FileBrowser:
    """A file/folder browser dialog."""

    def __init__(
        self,
        start_path: str = "",
        select_directory: bool = False,
        file_filter: Optional[List[str]] = None,
        title: str = "Select File",
        show_files_with_directory: bool = False,
    ):
        """Create a file browser.

        Args:
            start_path: Starting directory path
            select_directory: If True, select directories; otherwise select files
            file_filter: List of allowed extensions (e.g., ['.fasta', '.fa'])
            title: Dialog title
            show_files_with_directory: If True, show filtered files even in directory mode
        """
        self.select_directory = select_directory
        self.file_filter = file_filter
        self.title = title
        self.show_files_with_directory = show_files_with_directory
        self.selected_path: Optional[str] = None
        self._dialog: Optional[ui.dialog] = None
        self._path_display: Optional[ui.label] = None
        self._file_list: Optional[ui.column] = None

        # Determine starting path
        if start_path and Path(start_path).exists():
            if Path(start_path).is_file():
                self.current_path = Path(start_path).parent
            else:
                self.current_path = Path(start_path)
        else:
            self.current_path = Path.home()

    def _get_items(self) -> List[tuple]:
        """Get list of (name, path, is_dir) for current directory."""
        items = []

        try:
            for item in sorted(self.current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if item.name.startswith('.'):
                    continue  # Skip hidden files

                is_dir = item.is_dir()

                if is_dir:
                    items.append((item.name, str(item), True))
                elif not self.select_directory or self.show_files_with_directory:
                    # Show files if not in directory-only mode, or if show_files_with_directory is set
                    if self.file_filter:
                        if any(item.suffix.lower() == ext.lower() for ext in self.file_filter):
                            items.append((item.name, str(item), False))
                    else:
                        items.append((item.name, str(item), False))

        except PermissionError:
            pass

        return items

    def _update_file_list(self):
        """Update the file list display."""
        if self._file_list is None:
            return

        self._file_list.clear()

        with self._file_list:
            # Parent directory button
            if self.current_path.parent != self.current_path:
                with ui.row().classes("w-full items-center hover:bg-gray-100 cursor-pointer p-2 rounded").on(
                    "click", lambda: self._navigate_to(self.current_path.parent)
                ):
                    ui.icon("folder_open", color="amber").classes("text-xl")
                    ui.label("..").classes("ml-2")

            # List items
            for name, path, is_dir in self._get_items():
                with ui.row().classes("w-full items-center hover:bg-gray-100 cursor-pointer p-2 rounded").on(
                    "click", lambda p=path, d=is_dir: self._item_clicked(p, d)
                ):
                    if is_dir:
                        ui.icon("folder", color="amber").classes("text-xl")
                    else:
                        ui.icon("description", color="blue").classes("text-xl")
                    ui.label(name).classes("ml-2 truncate")

        if self._path_display:
            self._path_display.text = str(self.current_path)

    def _navigate_to(self, path: Path):
        """Navigate to a directory."""
        if path.is_dir():
            self.current_path = path
            self._update_file_list()

    def _item_clicked(self, path: str, is_dir: bool):
        """Handle item click."""
        p = Path(path)

        if is_dir:
            # Navigate into directory on click
            self._navigate_to(p)
        else:
            # File selected - submit immediately
            self.selected_path = path
            if self._dialog:
                self._dialog.submit(path)

    def _select_current_folder(self):
        """Select the current folder (for directory mode)."""
        self.selected_path = str(self.current_path)
        if self._dialog:
            self._dialog.submit(self.selected_path)

    async def pick(self) -> Optional[str]:
        """Show the file browser dialog and return selected path."""
        with ui.dialog() as self._dialog, ui.card().classes("w-[600px] h-[500px]"):
            # Header
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label(self.title).classes("text-lg font-bold")
                ui.button(icon="close", on_click=lambda: self._dialog.submit(None)).props("flat dense")

            # Path display with navigation
            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.button(icon="arrow_upward", on_click=lambda: self._navigate_to(self.current_path.parent)).props("flat dense")
                ui.button(icon="home", on_click=lambda: self._navigate_to(Path.home())).props("flat dense")
                self._path_display = ui.label(str(self.current_path)).classes("flex-grow text-sm font-mono bg-gray-100 p-2 rounded truncate")

            # Quick access buttons
            with ui.row().classes("w-full gap-2 mb-2"):
                ui.button("Home", on_click=lambda: self._navigate_to(Path.home())).props("flat dense")
                ui.button("Desktop", on_click=lambda: self._navigate_to(Path.home() / "Desktop")).props("flat dense")
                ui.button("Documents", on_click=lambda: self._navigate_to(Path.home() / "Documents")).props("flat dense")

            # File list
            with ui.scroll_area().classes("w-full flex-grow border rounded"):
                self._file_list = ui.column().classes("w-full")

            self._update_file_list()

            # Footer buttons
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=lambda: self._dialog.submit(None)).props("flat")

                if self.select_directory:
                    ui.button(
                        "Select This Folder",
                        on_click=self._select_current_folder,
                    ).props("color=primary")

        result = await self._dialog
        return result


def path_input(
    label: str,
    value: str,
    on_change: Callable[[str], None],
    placeholder: str = "",
    validation: Optional[Callable[[str], Optional[str]]] = None,
    directory: bool = False,
    file_filter: Optional[List[str]] = None,
) -> ui.input:
    """Create a path input with browse button.

    Args:
        label: Label for the input field
        value: Current value
        on_change: Callback when value changes
        placeholder: Placeholder text
        validation: Optional validation function returning error message or None
        directory: If True, browse for directories; otherwise for files
        file_filter: List of allowed file extensions

    Returns:
        The input element
    """

    def handle_change(e):
        new_value = e.value
        on_change(new_value)

    async def browse():
        browser = FileBrowser(
            start_path=input_field.value or str(Path.home()),
            select_directory=directory,
            file_filter=file_filter,
            title=f"Select {'Folder' if directory else 'File'}: {label}",
        )
        result = await browser.pick()
        if result:
            input_field.value = result
            on_change(result)

    with ui.row().classes("w-full items-end gap-2"):
        input_field = ui.input(
            label=label,
            value=value,
            placeholder=placeholder,
            on_change=handle_change,
        ).classes("flex-grow")

        ui.button(icon="folder_open", on_click=browse).props("flat").tooltip("Browse...")

        if validation:

            def validate(v):
                error = validation(v)
                return error if error else True

            input_field.validation = {"": validate}

    return input_field


def fasta_picker(
    value: str,
    on_change: Callable[[str], None],
) -> ui.input:
    """Create a FASTA file picker.

    Args:
        value: Current value
        on_change: Callback when value changes

    Returns:
        The input element
    """

    def validate_fasta(path: str) -> Optional[str]:
        if not path:
            return "FASTA file is required"
        p = Path(path)
        if not p.exists():
            return "File does not exist"
        if not (p.suffix.lower() in [".fasta", ".fa", ".faa"] or p.is_dir()):
            return "Must be a FASTA file (.fasta, .fa, .faa) or directory"
        return None

    return path_input(
        label="FASTA File or Directory",
        value=value,
        on_change=on_change,
        placeholder="/path/to/proteome.fasta",
        validation=validate_fasta,
        file_filter=[".fasta", ".fa", ".faa"],
    )


def reference_picker(
    value: str,
    on_change: Callable[[str], None],
) -> ui.input:
    """Create a reference dataset picker.

    Args:
        value: Current value
        on_change: Callback when value changes

    Returns:
        The input element
    """

    def validate_reference(path: str) -> Optional[str]:
        if not path:
            return "Reference dataset is required"
        p = Path(path)
        if not p.exists():
            return "Path does not exist"
        if not (p.suffix == ".d" and p.is_dir()):
            return "Must be a Bruker .d folder"
        return None

    return path_input(
        label="Reference Dataset (.d folder)",
        value=value,
        on_change=on_change,
        placeholder="/path/to/reference.d",
        validation=validate_reference,
        directory=True,  # .d folders are directories
    )


def output_picker(
    value: str,
    on_change: Callable[[str], None],
) -> ui.input:
    """Create an output directory picker.

    Args:
        value: Current value
        on_change: Callback when value changes

    Returns:
        The input element
    """

    def validate_output(path: str) -> Optional[str]:
        if not path:
            return "Output directory is required"
        return None

    return path_input(
        label="Output Directory",
        value=value,
        on_change=on_change,
        placeholder="/path/to/output",
        validation=validate_output,
        directory=True,
    )
