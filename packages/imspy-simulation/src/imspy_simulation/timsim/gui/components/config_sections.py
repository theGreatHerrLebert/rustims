"""Configuration section components for TimSim GUI."""

from typing import Callable, Optional
from nicegui import ui


class ConfigSection:
    """A collapsible configuration section with form fields."""

    def __init__(
        self,
        title: str,
        icon: str = "settings",
        expanded: bool = False,
    ):
        """Create a configuration section.

        Args:
            title: Section title
            icon: Material icon name
            expanded: Whether section is initially expanded
        """
        self.title = title
        self.icon = icon
        self.expanded = expanded
        self._expansion = None
        self._container = None

    def __enter__(self):
        """Enter context manager - create expansion panel."""
        self._expansion = ui.expansion(
            self.title,
            icon=self.icon,
            value=self.expanded,
        ).classes("w-full")
        self._expansion.__enter__()
        self._container = ui.column().classes("w-full gap-4 p-2")
        self._container.__enter__()
        return self

    def __exit__(self, *args):
        """Exit context manager."""
        self._container.__exit__(*args)
        self._expansion.__exit__(*args)


def number_field(
    label: str,
    value: float,
    on_change: Callable[[float], None],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    step: float = 1.0,
    suffix: str = "",
    tooltip: str = "",
) -> ui.number:
    """Create a number input field.

    Args:
        label: Field label
        value: Current value
        on_change: Callback when value changes
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        step: Step increment
        suffix: Suffix text (e.g., "ppm", "seconds")
        tooltip: Tooltip text

    Returns:
        The number input element
    """

    def handle_change(e):
        if e.value is not None:
            on_change(e.value)

    with ui.row().classes("w-full items-center gap-2"):
        field = ui.number(
            label=label,
            value=value,
            min=min_val,
            max=max_val,
            step=step,
            on_change=handle_change,
        ).classes("flex-grow")

        if suffix:
            ui.label(suffix).classes("text-gray-500")

        if tooltip:
            field.tooltip(tooltip)

    return field


def slider_field(
    label: str,
    value: float,
    on_change: Callable[[float], None],
    min_val: float = 0.0,
    max_val: float = 100.0,
    step: float = 1.0,
    format_fn: Optional[Callable[[float], str]] = None,
) -> ui.slider:
    """Create a slider with value display.

    Args:
        label: Field label
        value: Current value
        on_change: Callback when value changes
        min_val: Minimum value
        max_val: Maximum value
        step: Step increment
        format_fn: Function to format the displayed value

    Returns:
        The slider element
    """

    def handle_change(e):
        on_change(e.value)
        if value_label:
            val = format_fn(e.value) if format_fn else str(e.value)
            value_label.text = val

    ui.label(label).classes("font-medium")

    with ui.row().classes("w-full items-center gap-4"):
        slider = ui.slider(
            value=value,
            min=min_val,
            max=max_val,
            step=step,
            on_change=handle_change,
        ).classes("flex-grow")

        val = format_fn(value) if format_fn else str(value)
        value_label = ui.label(val).classes("min-w-16 text-right font-mono")

    return slider


def toggle_field(
    label: str,
    value: bool,
    on_change: Callable[[bool], None],
    tooltip: str = "",
) -> ui.switch:
    """Create a toggle switch.

    Args:
        label: Field label
        value: Current value
        on_change: Callback when value changes
        tooltip: Tooltip text

    Returns:
        The switch element
    """

    def handle_change(e):
        on_change(e.value)

    with ui.row().classes("w-full items-center justify-between"):
        lbl = ui.label(label)
        if tooltip:
            lbl.tooltip(tooltip)
        switch = ui.switch(value=value, on_change=handle_change)

    return switch


def select_field(
    label: str,
    value: str,
    options: list[str],
    on_change: Callable[[str], None],
    tooltip: str = "",
) -> ui.select:
    """Create a select dropdown.

    Args:
        label: Field label
        value: Current value
        options: List of options
        on_change: Callback when value changes
        tooltip: Tooltip text

    Returns:
        The select element
    """

    def handle_change(e):
        on_change(e.value)

    select = ui.select(
        options=options,
        value=value,
        label=label,
        on_change=handle_change,
    ).classes("w-full")

    if tooltip:
        select.tooltip(tooltip)

    return select


def text_field(
    label: str,
    value: str,
    on_change: Callable[[str], None],
    placeholder: str = "",
    tooltip: str = "",
) -> ui.input:
    """Create a text input field.

    Args:
        label: Field label
        value: Current value
        on_change: Callback when value changes
        placeholder: Placeholder text
        tooltip: Tooltip text

    Returns:
        The input element
    """

    def handle_change(e):
        on_change(e.value)

    field = ui.input(
        label=label,
        value=value,
        placeholder=placeholder,
        on_change=handle_change,
    ).classes("w-full")

    if tooltip:
        field.tooltip(tooltip)

    return field
