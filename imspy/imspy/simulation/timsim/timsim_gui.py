import sys
import time
from pathlib import Path

import markdown
import qdarkstyle

import toml
import numpy as np
from scipy.special import erfc
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, QSpinBox, QToolButton,
    QDoubleSpinBox, QComboBox, QGroupBox, QScrollArea, QAction, QMessageBox,
    QTextEdit, QSlider, QGridLayout, QTabWidget
)
from PyQt5.QtCore import Qt, QProcess, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import matplotlib.pyplot as plt

from imspy.timstof import TimsDatasetDIA
from imspy.timstof.data import TimsFrame, TimsSlice

# **VTKVisualizer Class**
class VTKVisualizer(QWidget):
    def __init__(self, data_path, parent=None):
        super(VTKVisualizer, self).__init__(parent)
        self.data_path = data_path

        self.layout = QVBoxLayout(self)

        # VTK Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtkWidget)

        # Set up the renderer
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # Load and visualize data
        self.load_and_visualize_data()

    def load_and_visualize_data(self):
        # Clear the renderer
        self.renderer.RemoveAllViewProps()

        # **Load data using TimsDatasetDIA**
        try:
            dia_data = TimsDatasetDIA("/media/hd02/data/sim/dia/timsim-gui/plain/TIMSIM-HeLa-G3600-25KP-PLAIN-DIA-FRAGT/TIMSIM-HeLa-G3600-25KP-PLAIN-DIA-FRAGT.d")
            t_slice = dia_data.get_tims_slice(dia_data.precursor_frames[500:1000]).filter(
                mz_min=500, mz_max=800, intensity_min=5, mobility_min=0.5, mobility_max=1.1)
            mz = t_slice.df.mz.values
            rt = t_slice.df.retention_time.values
            intensity = t_slice.df.intensity.values
            im = t_slice.df.mobility.values

            # Proceed with visualization
            self.visualize_data(rt, im, mz, intensity)
        except Exception as e:
            # Handle exceptions (e.g., invalid data path)
            QMessageBox.warning(self, "Error", f"Failed to load data:\n{e}")

    def visualize_data(self, rt, im, mz, intensity):
        # Store original min and max values for real axis labeling
        rt_min, rt_max = rt.min(), rt.max()
        im_min, im_max = im.min(), im.max()
        mz_min, mz_max = mz.min(), mz.max()

        # Normalize the dimensions to create a more cubic aspect ratio
        rt_norm = (rt - rt_min) / (rt_max - rt_min)
        im_norm = (im - im_min) / (im_max - im_min)
        mz_norm = (mz - mz_min) / (mz_max - mz_min)

        # Log scale the intensity for visualization
        log_intensity = np.log10(intensity)

        # Map the intensity values to an inferno color gradient
        cmap = plt.get_cmap("inferno")
        norm_log_intensity = (log_intensity - log_intensity.min()) / (log_intensity.max() - log_intensity.min())
        colors_inferno = (cmap(norm_log_intensity)[:, :3] * 255).astype(np.uint8)

        # Create VTK point cloud
        mapper = self.create_vtk_point_cloud(rt_norm, im_norm, mz_norm, colors_inferno)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.6)  # Set point opacity to 60%

        # Add the point cloud actor to the renderer
        self.renderer.AddActor(actor)

        # Define bounds for the cube axes (using normalized range)
        bounds = [0, 1, 0, 1, 0, 1]

        # Add cube axes around the data points
        cube_axes_actor = self.create_cube_axes(bounds, rt_min, rt_max, im_min, im_max, mz_min, mz_max)
        self.renderer.AddActor(cube_axes_actor)

        # Set light gray background
        self.renderer.SetBackground(0.9, 0.9, 0.9)

        # Center the camera on the data
        self.renderer.ResetCamera()

        # Initialize the interactor
        self.vtkWidget.GetRenderWindow().Render()

    def create_vtk_point_cloud(self, rt, im, mz, colors_inferno):
        # Create a VTK Points object
        points = vtk.vtkPoints()
        vtk_points_array = numpy_to_vtk(np.column_stack((rt, im, mz)).astype(np.float32), deep=True)
        points.SetData(vtk_points_array)

        # Convert the inferno colors to VTK array and add as colors
        vtk_colors = numpy_to_vtk(colors_inferno, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_colors.SetName("Colors")

        # Create a PolyData object to hold points and colors
        point_cloud = vtk.vtkPolyData()
        point_cloud.SetPoints(points)
        point_cloud.GetPointData().SetScalars(vtk_colors)  # Set the colors array for the mapper to use

        # Set up the mapper for rendering points
        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetInputData(point_cloud)
        mapper.ScalarVisibilityOn()  # Enable scalar coloring
        mapper.SetColorModeToDefault()  # Use the colors from the data

        # Define the glyph shape (e.g., spheres)
        glyph = vtk.vtkSphereSource()
        glyph.SetRadius(0.001)  # Reduced size for less overplotting
        mapper.SetSourceConnection(glyph.GetOutputPort())

        return mapper

    def create_cube_axes(self, bounds, rt_min, rt_max, im_min, im_max, mz_min, mz_max):
        axes = vtk.vtkCubeAxesActor()
        axes.SetBounds(bounds)
        axes.SetCamera(self.renderer.GetActiveCamera())

        # Set axis labels with real values
        axes.SetXTitle("Retention Time (s)")
        axes.SetYTitle("Ion Mobility (1/K0)")
        axes.SetZTitle("m/z")
        axes.SetXUnits(f"[{rt_min:.1f} - {rt_max:.1f}]")
        axes.SetYUnits(f"[{im_min:.1f} - {im_max:.1f}]")
        axes.SetZUnits(f"[{mz_min:.1f} - {mz_max:.1f}]")

        # Configure properties
        axes.GetTitleTextProperty(0).SetColor(0, 0, 0)  # X-axis title color
        axes.GetTitleTextProperty(1).SetColor(0, 0, 0)  # Y-axis title color
        axes.GetTitleTextProperty(2).SetColor(0, 0, 0)  # Z-axis title color
        axes.GetLabelTextProperty(0).SetColor(0, 0, 0)  # X-axis labels color
        axes.GetLabelTextProperty(1).SetColor(0, 0, 0)  # Y-axis labels color
        axes.GetLabelTextProperty(2).SetColor(0, 0, 0)  # Z-axis labels color

        # Enable grid lines for visual reference
        axes.DrawXGridlinesOn()
        axes.DrawYGridlinesOn()
        axes.DrawZGridlinesOn()
        axes.SetGridLineLocation(axes.VTK_GRID_LINES_FURTHEST)

        return axes


# CollapsibleBox class definition
class CollapsibleBox(QWidget):
    def __init__(self, title="", info_text="", parent=None):
        super().__init__(parent)

        # Toggle button for expanding/collapsing
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        # Info icon as a QLabel with hover tooltip
        self.info_label = QLabel()
        icon_path = Path(__file__).parent.parent / "resources/icons/info_icon.png"
        if icon_path.exists():
            self.info_label.setPixmap(QPixmap(str(icon_path)).scaled(19, 19, Qt.KeepAspectRatio))
            self.info_label.setToolTip(info_text)  # Set tooltip text for hover

        # Header layout with toggle button and info icon
        self.header_layout = QHBoxLayout()
        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.info_label)
        self.header_layout.addStretch()

        # Content area for collapsible content
        self.content_area = QWidget()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(25, 0, 0, 0)  # Indent content
        self.content_area.setLayout(self.content_layout)

        # Main layout to hold header and collapsible content
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(self.header_layout)
        main_layout.addWidget(self.content_area)
        self.setLayout(main_layout)

    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        # Expand or collapse content area
        if checked:
            self.content_area.setMaximumHeight(16777215)  # Arbitrary high max to expand fully
            self.content_area.setMinimumHeight(0)
        else:
            self.content_area.setMaximumHeight(0)
            self.content_area.setMinimumHeight(0)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


# EMGPlot class definition
class EMGPlot(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        # Fixed mean value at 0
        self.mean = 0
        # Default values for parameters (match with your spin boxes)
        self.std = 1.5  # Corresponds to Mean Std RT
        self.skewness = 0.3  # Corresponds to Mean Skewness
        # Variance values for randomness
        self.std_variance = 0.3  # Corresponds to Variance Std RT
        self.skewness_variance = 0.1  # Corresponds to Variance Skewness
        # Number of curves
        self.N = 25
        self.x = np.linspace(-20, 20, 1000)
        self.update_plot()

    def emg_pdf(self, x, mean, std, skewness):
        lambda_ = 1 / skewness if skewness != 0 else 1e6  # Avoid division by zero
        exp_component = lambda_ / 2 * np.exp(
            lambda_ / 2 * (2 * mean + lambda_ * std ** 2 - 2 * x)
        )
        erfc_component = erfc(
            (mean + lambda_ * std ** 2 - x) / (np.sqrt(2) * std)
        )
        return exp_component * erfc_component

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        for _ in range(self.N):
            sampled_std = max(np.random.normal(self.std, self.std_variance), 0.01)
            sampled_skewness = max(
                np.random.normal(self.skewness, self.skewness_variance), 0.01
            )
            y = self.emg_pdf(self.x, self.mean, sampled_std, sampled_skewness)
            self.ax.plot(self.x, y, alpha=0.5)
        self.ax.set_title("Exponentially Modified Gaussian Distribution")
        self.ax.set_xlabel("Retention Time Spread [Seconds]")
        self.ax.set_ylabel("Intensity")
        self.draw()

    # Methods to update parameters
    def set_std(self, value):
        self.std = value
        self.update_plot()

    def set_std_variance(self, value):
        self.std_variance = value
        self.update_plot()

    def set_skewness(self, value):
        self.skewness = value
        self.update_plot()

    def set_skewness_variance(self, value):
        self.skewness_variance = value
        self.update_plot()


# MainWindow class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Path to the script directory
        self.script_dir = Path(__file__).parent.parent / "resources/icons/"

        self.setWindowTitle("🦀💻 TimSim 🔬🐍 - Proteomics Experiment Simulation on timsTOF")
        self.setGeometry(100, 100, 800, 600)

        # Set the app logo
        self.setWindowIcon(QIcon(str(self.script_dir / "logo_2.png")))

        # Load the info icon image once
        self.info_icon = QPixmap(str(self.script_dir / "info_icon.png")).scaled(19, 19, Qt.KeepAspectRatio)

        # Initialize the menu bar
        self.init_menu_bar()

        # Create the tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        self.create_main_tab()
        self.create_documentation_tab()

        # Initialize the process variable
        self.process = None

    def create_documentation_tab(self):
        """Create a tab to display the manual."""
        documentation_tab = QWidget()
        layout = QVBoxLayout(documentation_tab)

        # Markdown file path
        doc_path = Path(__file__).parent.parent / "resources/docs/documentation.md"

        if doc_path.exists():
            try:
                # Read and convert Markdown to HTML
                with open(doc_path, "r", encoding="utf-8") as f:
                    markdown_text = f.read()
                html_content = markdown.markdown(markdown_text)
            except Exception as e:
                html_content = f"<p>Error loading documentation: {str(e)}</p>"
        else:
            html_content = "<p>Documentation file not found.</p>"

        # Render Markdown as HTML in a QTextEdit (basic support)
        self.documentation_viewer = QTextEdit()
        self.documentation_viewer.setHtml(html_content)
        self.documentation_viewer.setReadOnly(True)

        # Add to layout
        layout.addWidget(self.documentation_viewer)
        self.tab_widget.addTab(documentation_tab, "Documentation")

    def create_main_tab(self):
        # Create the main tab widget
        main_tab = QWidget()
        self.main_layout = QVBoxLayout(main_tab)

        # Initialize sections
        self.init_main_settings()
        self.init_peptide_digestion_settings()
        self.init_peptide_intensity_settings()  # Add this line
        self.init_isotopic_pattern_settings()
        self.init_distribution_settings()
        self.init_noise_settings()
        self.init_charge_state_probabilities()
        self.init_performance_settings()
        self.init_console()

        # Add all sections to the main layout
        self.main_layout.addWidget(self.main_settings_group)
        self.main_layout.addWidget(self.peptide_digestion_group)
        self.main_layout.addWidget(self.peptide_intensity_group)  # Add this line
        self.main_layout.addWidget(self.isotopic_pattern_group)
        self.main_layout.addWidget(self.distribution_settings_group)
        self.main_layout.addWidget(self.noise_settings_group)
        self.main_layout.addWidget(self.charge_state_probabilities_group)
        self.main_layout.addWidget(self.performance_settings_group)
        self.main_layout.addWidget(self.console)
        self.main_layout.addWidget(self.run_button)
        self.main_layout.addWidget(self.cancel_button)

        # Add spacing
        self.main_layout.addStretch()

        # Add logo at the bottom
        self.logo_label = QLabel()
        pixmap = QPixmap(str(self.script_dir / "rustims_logo.png"))
        scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(scaled_pixmap)
        self.logo_label.setFixedSize(200, 200)

        # Create a horizontal layout to center the logo label
        logo_layout = QHBoxLayout()
        logo_layout.addStretch()
        logo_layout.addWidget(self.logo_label)
        logo_layout.addStretch()

        # Add the centered logo layout to the main layout
        self.main_layout.addLayout(logo_layout)

        # Add scroll area if needed
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(main_tab)

        # Add the scroll area to the first tab
        self.tab_widget.addTab(scroll, "Main Settings")

    def init_menu_bar(self):
        menubar = self.menuBar()

        # Create 'File' menu
        file_menu = menubar.addMenu('File')

        # Add 'Save Config' action
        save_action = QAction('Save Config', self)
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)

        # Add 'Load Config' action
        load_action = QAction('Load Config', self)
        load_action.triggered.connect(self.load_config)
        file_menu.addAction(load_action)

    def add_setting_with_info(self, layout, label_text, widget, tooltip_text):
        setting_widget = QWidget()
        setting_layout = QHBoxLayout(setting_widget)
        setting_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(label_text)
        info_label = QLabel()
        info_label.setPixmap(self.info_icon)
        info_label.setToolTip(tooltip_text)
        setting_layout.addWidget(label)
        setting_layout.addWidget(widget)
        setting_layout.addWidget(info_label)
        layout.addWidget(setting_widget)
        return setting_widget  # Return the setting widget to control its visibility

    def add_spinbox_with_info(self, layout, attribute_name, label_text, tooltip_text, min_value, max_value,
                              default_value, decimals=2, step=None):
        setting_widget = QWidget()
        setting_layout = QHBoxLayout(setting_widget)
        setting_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(label_text)
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_value, max_value)
        spinbox.setDecimals(decimals)
        if step is not None:
            spinbox.setSingleStep(step)
        spinbox.setValue(default_value)
        info_label = QLabel()
        info_label.setPixmap(self.info_icon)
        info_label.setToolTip(tooltip_text)
        setting_layout.addWidget(label)
        setting_layout.addWidget(spinbox)
        setting_layout.addWidget(info_label)
        layout.addWidget(setting_widget)
        setattr(self, attribute_name, spinbox)  # Assign spinbox to self.attribute_name

    def add_checkbox_with_info(self, layout, checkbox, tooltip_text):
        checkbox_layout = QHBoxLayout()
        info_label = QLabel()
        info_label.setPixmap(self.info_icon)
        info_label.setToolTip(tooltip_text)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.addWidget(info_label)
        layout.addLayout(checkbox_layout)

    def init_main_settings(self):
        self.main_settings_group = QGroupBox("Main Settings")
        layout = QVBoxLayout()

        # Save Path
        self.path_input = QLineEdit()
        self.path_browse = QPushButton("Browse")
        self.path_browse.clicked.connect(self.browse_save_path)
        path_widget = QWidget()
        path_layout = QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.path_browse)
        self.add_setting_with_info(
            layout,
            "Save Path:",
            path_widget,
            "Select the directory to save simulation outputs."
        )

        # Reference Dataset Path
        self.reference_input = QLineEdit()
        self.reference_browse = QPushButton("Browse")
        self.reference_browse.clicked.connect(self.browse_reference_path)
        reference_widget = QWidget()
        reference_layout = QHBoxLayout(reference_widget)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.addWidget(self.reference_input)
        reference_layout.addWidget(self.reference_browse)
        self.add_setting_with_info(
            layout,
            "Reference Dataset Path:",
            reference_widget,
            "Specify the path to the reference dataset used as a template for simulation."
        )

        # FASTA File Path
        self.fasta_input = QLineEdit()
        self.fasta_browse = QPushButton("Browse")
        self.fasta_browse.clicked.connect(self.browse_fasta_path)
        fasta_widget = QWidget()
        fasta_layout = QHBoxLayout(fasta_widget)
        fasta_layout.setContentsMargins(0, 0, 0, 0)
        fasta_layout.addWidget(self.fasta_input)
        fasta_layout.addWidget(self.fasta_browse)
        self.add_setting_with_info(
            layout,
            "FASTA File Path:",
            fasta_widget,
            "Provide the path to the FASTA file containing protein sequences for digestion."
        )

        # Experiment Name
        self.name_input = QLineEdit(f"TIMSIM-{int(time.time())}")
        self.add_setting_with_info(
            layout,
            "Experiment Name:",
            self.name_input,
            "Name your experiment to identify it in saved outputs and reports."
        )

        # Acquisition Type
        self.acquisition_combo = QComboBox()
        self.acquisition_combo.addItems(["DIA", "SYNCHRO", "SLICE", "MIDIA"])
        self.add_setting_with_info(
            layout,
            "Acquisition Type:",
            self.acquisition_combo,
            "Select the acquisition method used to simulate data."
        )

        # Options with checkboxes and icons to the right
        options_layout = QVBoxLayout()

        # Use Reference Layout
        self.use_reference_layout_checkbox = QCheckBox("Use Reference Layout")
        self.use_reference_layout_checkbox.setChecked(True)
        self.add_checkbox_with_info(
            options_layout,
            self.use_reference_layout_checkbox,
            "Use the layout of the reference dataset as a template for simulation."
        )

        # Load Reference into Memory
        self.reference_in_memory_checkbox = QCheckBox("Load Reference into Memory")
        self.reference_in_memory_checkbox.setChecked(False)
        self.add_checkbox_with_info(
            options_layout,
            self.reference_in_memory_checkbox,
            "Load the reference dataset entirely into memory to speed up access."
        )

        # Sample Peptides
        self.sample_peptides_checkbox = QCheckBox("Sample Peptides")
        self.sample_peptides_checkbox.setChecked(True)
        self.add_checkbox_with_info(
            options_layout,
            self.sample_peptides_checkbox,
            "Enable to randomly sample peptides for a subset of the dataset."
        )

        # Sample Seed
        self.sample_seed_spin = QSpinBox()
        self.sample_seed_spin.setRange(0, 1000000)
        self.sample_seed_spin.setValue(41)
        self.add_setting_with_info(
            options_layout,
            "Sample Seed:",
            self.sample_seed_spin,
            "Set the seed for the random peptide sampling."
        )

        # Generate Decoys
        self.add_decoys_checkbox = QCheckBox("Generate Decoys")
        self.add_decoys_checkbox.setChecked(False)
        self.add_checkbox_with_info(
            options_layout,
            self.add_decoys_checkbox,
            "Also simulate inverted decoy peptides (non-biological)."
        )

        # Silent Mode
        self.silent_checkbox = QCheckBox("Silent Mode")
        self.silent_checkbox.setChecked(False)
        self.add_checkbox_with_info(
            options_layout,
            self.silent_checkbox,
            "Run the simulation in silent mode, suppressing most output messages."
        )

        # Apply Fragmentation to Ions
        self.apply_fragmentation_checkbox = QCheckBox("Apply Fragmentation to Ions")
        self.apply_fragmentation_checkbox.setChecked(True)
        self.add_checkbox_with_info(
            options_layout,
            self.apply_fragmentation_checkbox,
            "If disabled, quadrupole selection will still be applied but fragmentation will be skipped."
        )

        # Use Existing Data as Template
        self.from_existing_checkbox = QCheckBox("Use existing simulated data as template")
        self.from_existing_checkbox.setChecked(False)
        self.add_checkbox_with_info(
            options_layout,
            self.from_existing_checkbox,
            "Use existing simulated data as template for the current simulation."
        )

        # Proteome Mixture
        self.proteome_mix_checkbox = QCheckBox("Proteome Mixture")
        self.proteome_mix_checkbox.setChecked(False)
        self.add_checkbox_with_info(
            options_layout,
            self.proteome_mix_checkbox,
            "Use a mixture of proteomes to simulate a more complex sample."
        )

        layout.addLayout(options_layout)

        # Existing simulated Experiment path (visible based on checkbox)
        self.existing_path_input = QLineEdit()
        self.existing_path_browse = QPushButton("Browse")
        self.existing_path_browse.clicked.connect(self.browse_existing_save_path)
        existing_path_widget = QWidget()
        existing_path_layout = QHBoxLayout(existing_path_widget)
        existing_path_layout.setContentsMargins(0, 0, 0, 0)
        existing_path_layout.addWidget(self.existing_path_input)
        existing_path_layout.addWidget(self.existing_path_browse)
        existing_setting_widget = self.add_setting_with_info(
            layout,
            "Existing simulated Experiment:",
            existing_path_widget,
            "Select the directory where the existing simulated data is saved."
        )
        # Hide the entire setting initially
        existing_setting_widget.setVisible(False)
        self.from_existing_checkbox.toggled.connect(existing_setting_widget.setVisible)

        # Multi-Fasta Path (visible based on checkbox)
        self.multi_fasta_input = QLineEdit()
        self.multi_fasta_browse = QPushButton("Browse")
        self.multi_fasta_browse.clicked.connect(self.browse_multi_fasta_path)
        multi_fasta_widget = QWidget()
        multi_fasta_layout = QHBoxLayout(multi_fasta_widget)
        multi_fasta_layout.setContentsMargins(0, 0, 0, 0)
        multi_fasta_layout.addWidget(self.multi_fasta_input)
        multi_fasta_layout.addWidget(self.multi_fasta_browse)
        multi_fasta_setting_widget = self.add_setting_with_info(
            layout,
            "Multi-Fasta Path:",
            multi_fasta_widget,
            "Select the directory containing multiple FASTA files for the proteome mixture."
        )
        # Hide the entire setting initially
        multi_fasta_setting_widget.setVisible(False)

        # Multi-Fasta Dilution Factor Path (visible based on checkbox)
        self.multi_fasta_dilution_input = QLineEdit()
        self.multi_fasta_dilution_browse = QPushButton("Browse")
        self.multi_fasta_dilution_browse.clicked.connect(self.browse_multi_fasta_dilution_path)
        multi_fasta_dilution_widget = QWidget()
        multi_fasta_dilution_layout = QHBoxLayout(multi_fasta_dilution_widget)
        multi_fasta_dilution_layout.setContentsMargins(0, 0, 0, 0)
        multi_fasta_dilution_layout.addWidget(self.multi_fasta_dilution_input)
        multi_fasta_dilution_layout.addWidget(self.multi_fasta_dilution_browse)
        multi_fasta_dilution_setting_widget = self.add_setting_with_info(
            layout,
            "Multi-Fasta Dilution Factor Path:",
            multi_fasta_dilution_widget,
            "Select the CSV file containing the dilution factors for the proteome mixture."
        )
        # Hide the entire setting initially
        multi_fasta_dilution_setting_widget.setVisible(False)

        # Connect the visibility of multi-fasta widgets to the proteome_mix_checkbox
        self.proteome_mix_checkbox.toggled.connect(multi_fasta_setting_widget.setVisible)
        self.proteome_mix_checkbox.toggled.connect(multi_fasta_dilution_setting_widget.setVisible)

        # Set layout for the group box
        self.main_settings_group.setLayout(layout)
        self.main_layout.addWidget(self.main_settings_group)

    def init_peptide_digestion_settings(self):
        info_text = "Configure how peptides are generated from protein sequences, including cleavage rules and peptide length constraints."
        self.peptide_digestion_group = CollapsibleBox("Peptide Digestion Settings", info_text)
        layout = self.peptide_digestion_group.content_layout

        # Number of Sampled Peptides
        self.num_sample_peptides_spin = QSpinBox()
        self.num_sample_peptides_spin.setRange(1, 1000000)
        self.num_sample_peptides_spin.setValue(25000)
        self.add_setting_with_info(
            layout,
            "Number of Sampled Peptides:",
            self.num_sample_peptides_spin,
            "Total number of peptides to sample in the simulation."
        )

        # Missed Cleavages
        self.missed_cleavages_spin = QSpinBox()
        self.missed_cleavages_spin.setRange(0, 10)
        self.missed_cleavages_spin.setValue(2)
        self.add_setting_with_info(
            layout,
            "Missed Cleavages:",
            self.missed_cleavages_spin,
            "The number of allowed missed cleavages in peptides during digestion."
        )

        # Minimum Peptide Length
        self.min_len_spin = QSpinBox()
        self.min_len_spin.setRange(1, 100)
        self.min_len_spin.setValue(7)
        self.add_setting_with_info(
            layout,
            "Minimum Peptide Length:",
            self.min_len_spin,
            "Specifies the minimum length of peptides in amino acids."
        )

        # Maximum Peptide Length
        self.max_len_spin = QSpinBox()
        self.max_len_spin.setRange(1, 100)
        self.max_len_spin.setValue(30)
        self.add_setting_with_info(
            layout,
            "Maximum Peptide Length:",
            self.max_len_spin,
            "Specifies the maximum length of peptides in amino acids."
        )

        # Cleave At
        self.cleave_at_input = QLineEdit("KR")
        self.add_setting_with_info(
            layout,
            "Cleave At:",
            self.cleave_at_input,
            "Residues at which cleavage occurs, e.g., 'KR' for trypsin digestion."
        )

        # Restrict
        self.restrict_input = QLineEdit("P")
        self.add_setting_with_info(
            layout,
            "Restrict:",
            self.restrict_input,
            "Residues where cleavage is restricted, e.g., 'P' to prevent cleavage after proline."
        )

        # Modifications
        self.mods_input = QLineEdit()
        self.mods_browse = QPushButton("Browse")
        self.mods_browse.clicked.connect(self.load_modifications)
        mods_widget = QWidget()
        mods_layout = QHBoxLayout(mods_widget)
        mods_layout.setContentsMargins(0, 0, 0, 0)
        mods_layout.addWidget(self.mods_input)
        mods_layout.addWidget(self.mods_browse)
        self.add_setting_with_info(
            layout,
            "Amino Acid Modifications:",
            mods_widget,
            "Specify the path to the modifications TOML file, containing fixed and variable modifications."
        )

        # Add the updated collapsible group to the main layout
        self.main_layout.addWidget(self.peptide_digestion_group)

    def init_peptide_intensity_settings(self):
        info_text = "Set the intensity distribution parameters for peptides, including mean, min, max intensities."
        self.peptide_intensity_group = CollapsibleBox("Peptide Intensity Settings", info_text)
        layout = self.peptide_intensity_group.content_layout

        # Load info icon image
        info_icon = self.info_icon

        # Function to format slider value as 10^power
        def update_label_from_slider(slider, label):
            power = slider.value()
            label.setText(f"10^{power} ({10 ** power:.1e})")

        # Mean Intensity
        self.intensity_mean_label = QLabel()
        self.intensity_mean_slider = QSlider(Qt.Horizontal)
        self.intensity_mean_slider.setRange(0, 10)  # Range from 10^0 to 10^10
        self.intensity_mean_slider.setValue(7)  # Default to 10^7
        update_label_from_slider(self.intensity_mean_slider, self.intensity_mean_label)  # Initialize label

        self.intensity_mean_slider.valueChanged.connect(
            lambda: update_label_from_slider(self.intensity_mean_slider, self.intensity_mean_label)
        )
        layout.addWidget(QLabel("Mean Intensity:"))
        layout.addWidget(self.intensity_mean_label)
        layout.addWidget(self.intensity_mean_slider)

        # Mean Intensity Info Icon
        self.intensity_mean_info = QLabel()
        self.intensity_mean_info.setPixmap(info_icon.scaled(19, 19))
        self.intensity_mean_info.setToolTip(
            "The Mean Intensity represents the average intensity (1/lambda) for the simulated peptides, expressed as a power of 10.")
        layout.addWidget(self.intensity_mean_info)

        # Minimum Intensity
        self.intensity_min_label = QLabel()
        self.intensity_min_slider = QSlider(Qt.Horizontal)
        self.intensity_min_slider.setRange(0, 10)
        self.intensity_min_slider.setValue(5)  # Default to 10^5
        update_label_from_slider(self.intensity_min_slider, self.intensity_min_label)

        self.intensity_min_slider.valueChanged.connect(
            lambda: update_label_from_slider(self.intensity_min_slider, self.intensity_min_label)
        )
        layout.addWidget(QLabel("Minimum Intensity:"))
        layout.addWidget(self.intensity_min_label)
        layout.addWidget(self.intensity_min_slider)

        # Minimum Intensity Info Icon
        self.intensity_min_info = QLabel()
        self.intensity_min_info.setPixmap(info_icon.scaled(19, 19))
        self.intensity_min_info.setToolTip(
            "The Minimum Intensity is the lowest sampled amount of total peptide intensity, also expressed as a power of 10.")
        layout.addWidget(self.intensity_min_info)

        # Maximum Intensity
        self.intensity_max_label = QLabel()
        self.intensity_max_slider = QSlider(Qt.Horizontal)
        self.intensity_max_slider.setRange(0, 10)
        self.intensity_max_slider.setValue(9)  # Default to 10^9
        update_label_from_slider(self.intensity_max_slider, self.intensity_max_label)

        self.intensity_max_slider.valueChanged.connect(
            lambda: update_label_from_slider(self.intensity_max_slider, self.intensity_max_label)
        )
        layout.addWidget(QLabel("Maximum Intensity:"))
        layout.addWidget(self.intensity_max_label)
        layout.addWidget(self.intensity_max_slider)

        # Maximum Intensity Info Icon
        self.intensity_max_info = QLabel()
        self.intensity_max_info.setPixmap(info_icon.scaled(19, 19))
        self.intensity_max_info.setToolTip(
            "The Maximum Intensity defines the highest possible intensity for peptides in the dataset, expressed as a power of 10.")
        layout.addWidget(self.intensity_max_info)

        # Sample Occurrences Randomly
        self.sample_occurrences_checkbox = QCheckBox("Sample Occurrences Randomly")
        self.sample_occurrences_checkbox.setChecked(True)
        layout.addWidget(self.sample_occurrences_checkbox)

        # Sample Occurrences Info Icon
        self.sample_occurrences_info = QLabel()
        self.sample_occurrences_info.setPixmap(info_icon.scaled(19, 19))
        self.sample_occurrences_info.setToolTip(
            "If checked, peptide occurrences will be sampled randomly from a clipped exponential distribution with above specified intensity properties.")
        layout.addWidget(self.sample_occurrences_info)

        # Fixed Intensity Value
        self.intensity_value_label = QLabel()
        self.intensity_value_slider = QSlider(Qt.Horizontal)
        self.intensity_value_slider.setRange(0, 10)
        self.intensity_value_slider.setValue(6)  # Default to 10^6
        update_label_from_slider(self.intensity_value_slider, self.intensity_value_label)

        self.intensity_value_slider.valueChanged.connect(
            lambda: update_label_from_slider(self.intensity_value_slider, self.intensity_value_label)
        )
        layout.addWidget(QLabel("Fixed Intensity Value:"))
        layout.addWidget(self.intensity_value_label)
        layout.addWidget(self.intensity_value_slider)

        # Fixed Intensity Info Icon
        self.intensity_value_info = QLabel()
        self.intensity_value_info.setPixmap(info_icon.scaled(19, 19))
        self.intensity_value_info.setToolTip(
            "Set a fixed intensity value (as a power of 10) for all peptides when random sampling is disabled.")
        layout.addWidget(self.intensity_value_info)

    def init_isotopic_pattern_settings(self):
        info_text = "Configure the isotopic pattern generation parameters, including the number of isotopes and intensity thresholds."
        self.isotopic_pattern_group = CollapsibleBox("Isotopic Pattern Settings", info_text)
        layout = self.isotopic_pattern_group.content_layout

        # Load info icon image
        info_icon = QPixmap(str(self.script_dir / "info_icon.png")).scaled(19, 19, Qt.KeepAspectRatio)

        # Number of Isotopes
        self.isotope_k_label = QLabel("Maximum number of Isotopes:")
        self.isotope_k_spin = QSpinBox()
        self.isotope_k_spin.setRange(1, 20)
        self.isotope_k_spin.setValue(8)
        layout.addWidget(self.isotope_k_label)
        layout.addWidget(self.isotope_k_spin)

        # Number of Isotopes Info Icon
        self.isotope_k_info = QLabel()
        self.isotope_k_info.setPixmap(info_icon)
        self.isotope_k_info.setToolTip(
            "Specify the maximum number of isotopes to simulate for each peptide. Higher numbers increase simulation complexity.")
        layout.addWidget(self.isotope_k_info)

        # Minimum Isotope Intensity
        self.isotope_min_intensity_label = QLabel("Minimum Isotope Intensity:")
        self.isotope_min_intensity_spin = QSpinBox()
        self.isotope_min_intensity_spin.setRange(0, 1000)
        self.isotope_min_intensity_spin.setValue(1)
        layout.addWidget(self.isotope_min_intensity_label)
        layout.addWidget(self.isotope_min_intensity_spin)

        # Minimum Isotope Intensity Info Icon
        self.isotope_min_intensity_info = QLabel()
        self.isotope_min_intensity_info.setPixmap(info_icon)
        self.isotope_min_intensity_info.setToolTip(
            "Set the threshold intensity for the smallest isotope in the pattern to still be considered in the simulation.")
        layout.addWidget(self.isotope_min_intensity_info)

        # Centroid Isotopes
        self.isotope_centroid_checkbox = QCheckBox("Centroid Isotopes")
        self.isotope_centroid_checkbox.setChecked(True)
        layout.addWidget(self.isotope_centroid_checkbox)

        # Centroid Isotopes Info Icon
        self.isotope_centroid_info = QLabel()
        self.isotope_centroid_info.setPixmap(info_icon)
        self.isotope_centroid_info.setToolTip(
            "Enable to centroid isotopes, averaging peak positions for a simplified pattern (should be kept True since timsTOF centroids peaks).")
        layout.addWidget(self.isotope_centroid_info)

        # Add isotopic pattern group to main layout
        self.main_layout.addWidget(self.isotopic_pattern_group)

    def init_distribution_settings(self):
        info_text = "Adjust the parameters for signal distribution in the simulation, including gradient length and skewness."
        self.distribution_settings_group = CollapsibleBox("Signal Distribution Settings", info_text)

        # Create a horizontal layout to hold settings and plot side by side
        main_layout = QHBoxLayout()

        # Left vertical layout for settings
        settings_layout = QVBoxLayout()

        # Right vertical layout for the plot
        plot_layout = QVBoxLayout()

        # Define your settings in a list of dictionaries
        settings = [
            {
                'attribute_name': 'gradient_length_spin',
                'label_text': 'Gradient Length (seconds):',
                'tooltip_text': 'Defines the total length of the simulated chromatographic gradient in seconds.',
                'min_value': 1,
                'max_value': 1e5,
                'default_value': 3600,
                'decimals': 0
            },
            {
                'attribute_name': 'mean_std_rt_spin',
                'label_text': 'Mean Std RT:',
                'tooltip_text': 'The mean standard deviation for retention time, affecting peak widths.',
                'min_value': 0.1,
                'max_value': 10,
                'default_value': 1.0,
                'decimals': 2
            },
            {
                'attribute_name': 'variance_std_rt_spin',
                'label_text': 'Variance Std RT:',
                'tooltip_text': 'Variance of the standard deviation of retention time, affecting variance of distribution spread.',
                'min_value': 0,
                'max_value': 5,
                'default_value': 0.1,
                'decimals': 2
            },
            {
                'attribute_name': 'mean_skewness_spin',
                'label_text': 'Mean Skewness:',
                'tooltip_text': 'Average skewness for retention time distribution, impacting peak asymmetry.',
                'min_value': 0,
                'max_value': 5,
                'default_value': 1.5,
                'decimals': 2
            },
            {
                'attribute_name': 'variance_skewness_spin',
                'label_text': 'Variance Skewness:',
                'tooltip_text': 'Variance in skewness of retention time distribution, altering peak shapes.',
                'min_value': 0,
                'max_value': 5,
                'default_value': 0.1,
                'decimals': 2
            },
            {
                'attribute_name': 'std_im_spin',
                'label_text': 'Standard Deviation IM:',
                'tooltip_text': 'Standard deviation in ion mobility, which affects peak width in the mobility dimension.',
                'min_value': 0,
                'max_value': 1,
                'default_value': 0.01,
                'decimals': 4
            },
            {
                'attribute_name': 'variance_std_im_spin',
                'label_text': 'Variance Std IM:',
                'tooltip_text': 'Variance of the standard deviation of ion mobility, affecting distribution spread.',
                'min_value': 0,
                'max_value': 1,
                'default_value': 0.001,
                'decimals': 4
            },
            {
                'attribute_name': 'z_score_spin',
                'label_text': 'Z-Score:',
                'tooltip_text': 'Defines the z-score threshold for filtering data, removing low-signal regions.',
                'min_value': 0,
                'max_value': 5,
                'default_value': 0.99,
                'decimals': 2
            },
            {
                'attribute_name': 'target_p_spin',
                'label_text': 'Target Percentile:',
                'tooltip_text': 'Specifies the target percentile for retention time, capturing high-density regions.',
                'min_value': 0,
                'max_value': 1,
                'default_value': 0.999,
                'decimals': 3
            },
            {
                'attribute_name': 'sampling_step_size_spin',
                'label_text': 'Sampling Step Size:',
                'tooltip_text': 'Sets the step size for data sampling, adjusting resolution and computation time.',
                'min_value': 0,
                'max_value': 1,
                'default_value': 0.001,
                'decimals': 4
            },
        ]

        # Loop over the settings and add them to the layout
        for setting in settings:
            self.add_spinbox_with_info(
                layout=settings_layout,
                attribute_name=setting['attribute_name'],
                label_text=setting['label_text'],
                tooltip_text=setting['tooltip_text'],
                min_value=setting['min_value'],
                max_value=setting['max_value'],
                default_value=setting['default_value'],
                decimals=setting.get('decimals', 2),
                step=setting.get('step', None)
            )

        # Add the EMG visualization to the plot layout
        self.emg_plot = EMGPlot()
        plot_layout.addWidget(self.emg_plot)

        # Add settings and plot layouts to the main horizontal layout
        main_layout.addLayout(settings_layout)
        main_layout.addLayout(plot_layout)

        # Set the main layout to the content layout of the collapsible group
        self.distribution_settings_group.content_layout.addLayout(main_layout)

        # Connect the spin boxes to the EMGPlot methods
        self.mean_std_rt_spin.valueChanged.connect(self.update_emg_std)
        self.variance_std_rt_spin.valueChanged.connect(self.update_emg_std_variance)
        self.mean_skewness_spin.valueChanged.connect(self.update_emg_skewness)
        self.variance_skewness_spin.valueChanged.connect(self.update_emg_skewness_variance)

        # Add the distribution settings group to the main layout
        self.main_layout.addWidget(self.distribution_settings_group)

    # Methods to update EMGPlot parameters
    def update_emg_std(self, value):
        self.emg_plot.set_std(value)

    def update_emg_std_variance(self, value):
        self.emg_plot.set_std_variance(value)

    def update_emg_skewness(self, value):
        self.emg_plot.set_skewness(value)

    def update_emg_skewness_variance(self, value):
        self.emg_plot.set_skewness_variance(value)

    def init_noise_settings(self):
        info_text = "Configure the noise parameters for the simulation, including adding m/z noise and real data noise."
        self.noise_settings_group = CollapsibleBox("Noise Settings", info_text)
        layout = self.noise_settings_group.content_layout

        # Load info icon image
        info_icon = QPixmap(str(self.script_dir / "info_icon.png")).scaled(19, 19, Qt.KeepAspectRatio)

        # Add Noise to Signals
        noise_signals_layout = QHBoxLayout()
        self.add_noise_to_signals_checkbox = QCheckBox("Add Noise to Signals")
        self.add_noise_to_signals_checkbox.setChecked(True)
        noise_signals_layout.addWidget(self.add_noise_to_signals_checkbox)
        self.add_noise_to_signals_info = QLabel()
        self.add_noise_to_signals_info.setPixmap(info_icon)
        self.add_noise_to_signals_info.setToolTip("Enable or disable adding random noise to signal intensities.")
        noise_signals_layout.addWidget(self.add_noise_to_signals_info)
        layout.addLayout(noise_signals_layout)

        # Add Precursor M/Z Noise
        mz_noise_precursor_layout = QHBoxLayout()
        self.mz_noise_precursor_checkbox = QCheckBox("Add Precursor M/Z Noise")
        self.mz_noise_precursor_checkbox.setChecked(True)
        mz_noise_precursor_layout.addWidget(self.mz_noise_precursor_checkbox)
        self.mz_noise_precursor_info = QLabel()
        self.mz_noise_precursor_info.setPixmap(info_icon)
        self.mz_noise_precursor_info.setToolTip("Add random noise to precursor m/z values.")
        mz_noise_precursor_layout.addWidget(self.mz_noise_precursor_info)
        layout.addLayout(mz_noise_precursor_layout)

        # Precursor Noise PPM
        precursor_noise_ppm_layout = QHBoxLayout()
        self.precursor_noise_ppm_label = QLabel("Precursor Noise PPM:")
        self.precursor_noise_ppm_spin = QDoubleSpinBox()
        self.precursor_noise_ppm_spin.setRange(0, 100)
        self.precursor_noise_ppm_spin.setValue(5.0)
        precursor_noise_ppm_layout.addWidget(self.precursor_noise_ppm_label)
        precursor_noise_ppm_layout.addWidget(self.precursor_noise_ppm_spin)
        self.precursor_noise_ppm_info = QLabel()
        self.precursor_noise_ppm_info.setPixmap(info_icon)
        self.precursor_noise_ppm_info.setToolTip("Set the noise level (in parts per million) for precursor m/z values.")
        precursor_noise_ppm_layout.addWidget(self.precursor_noise_ppm_info)
        layout.addLayout(precursor_noise_ppm_layout)

        # Add Fragment M/Z Noise
        mz_noise_fragment_layout = QHBoxLayout()
        self.mz_noise_fragment_checkbox = QCheckBox("Add Fragment M/Z Noise")
        self.mz_noise_fragment_checkbox.setChecked(True)
        mz_noise_fragment_layout.addWidget(self.mz_noise_fragment_checkbox)
        self.mz_noise_fragment_info = QLabel()
        self.mz_noise_fragment_info.setPixmap(info_icon)
        self.mz_noise_fragment_info.setToolTip("Add random noise to fragment m/z values.")
        mz_noise_fragment_layout.addWidget(self.mz_noise_fragment_info)
        layout.addLayout(mz_noise_fragment_layout)

        # Fragment Noise PPM
        fragment_noise_ppm_layout = QHBoxLayout()
        self.fragment_noise_ppm_label = QLabel("Fragment Noise PPM:")
        self.fragment_noise_ppm_spin = QDoubleSpinBox()
        self.fragment_noise_ppm_spin.setRange(0, 100)
        self.fragment_noise_ppm_spin.setValue(5.0)
        fragment_noise_ppm_layout.addWidget(self.fragment_noise_ppm_label)
        fragment_noise_ppm_layout.addWidget(self.fragment_noise_ppm_spin)
        self.fragment_noise_ppm_info = QLabel()
        self.fragment_noise_ppm_info.setPixmap(info_icon)
        self.fragment_noise_ppm_info.setToolTip("Set the noise level (in parts per million) for fragment m/z values.")
        fragment_noise_ppm_layout.addWidget(self.fragment_noise_ppm_info)
        layout.addLayout(fragment_noise_ppm_layout)

        # Use Uniform Distribution
        mz_noise_uniform_layout = QHBoxLayout()
        self.mz_noise_uniform_checkbox = QCheckBox("Use Uniform Distribution for M/Z Noise")
        self.mz_noise_uniform_checkbox.setChecked(False)
        mz_noise_uniform_layout.addWidget(self.mz_noise_uniform_checkbox)
        self.mz_noise_uniform_info = QLabel()
        self.mz_noise_uniform_info.setPixmap(info_icon)
        self.mz_noise_uniform_info.setToolTip(
            "If enabled, use a uniform distribution instead of Gaussian for m/z noise.")
        mz_noise_uniform_layout.addWidget(self.mz_noise_uniform_info)
        layout.addLayout(mz_noise_uniform_layout)

        # Add Real Data Noise
        real_data_noise_layout = QHBoxLayout()
        self.add_real_data_noise_checkbox = QCheckBox("Add Real Data Noise")
        self.add_real_data_noise_checkbox.setChecked(True)
        real_data_noise_layout.addWidget(self.add_real_data_noise_checkbox)
        self.add_real_data_noise_info = QLabel()
        self.add_real_data_noise_info.setPixmap(info_icon)
        self.add_real_data_noise_info.setToolTip("Add noise derived from real data samples.")
        real_data_noise_layout.addWidget(self.add_real_data_noise_info)
        layout.addLayout(real_data_noise_layout)

        # Reference Noise Intensity Max
        reference_noise_intensity_layout = QHBoxLayout()
        self.reference_noise_intensity_max_label = QLabel("Reference Noise Intensity Max:")
        self.reference_noise_intensity_max_spin = QDoubleSpinBox()
        self.reference_noise_intensity_max_spin.setRange(1, 1e5)
        self.reference_noise_intensity_max_spin.setValue(75)
        reference_noise_intensity_layout.addWidget(self.reference_noise_intensity_max_label)
        reference_noise_intensity_layout.addWidget(self.reference_noise_intensity_max_spin)
        self.reference_noise_intensity_info = QLabel()
        self.reference_noise_intensity_info.setPixmap(info_icon)
        self.reference_noise_intensity_info.setToolTip(
            "Set the maximum noise intensity for reference data samples."
        )
        reference_noise_intensity_layout.addWidget(self.reference_noise_intensity_info)
        layout.addLayout(reference_noise_intensity_layout)

        # Downsample Factor
        downsample_factor_layout = QHBoxLayout()
        self.down_sample_factor_label = QLabel("Fragment downsample Factor:")
        self.down_sample_factor_spin = QDoubleSpinBox()
        self.down_sample_factor_spin.setRange(0, 1)
        self.down_sample_factor_spin.setDecimals(2)
        self.down_sample_factor_spin.setValue(0.5)
        downsample_factor_layout.addWidget(self.down_sample_factor_label)
        downsample_factor_layout.addWidget(self.down_sample_factor_spin)
        self.down_sample_factor_info = QLabel()
        self.down_sample_factor_info.setPixmap(info_icon)
        self.down_sample_factor_info.setToolTip("Set the downsample factor for fragment ions, fragment ions will be de-selected with probability inverse proportional to their intensity until the specified factor is reached.")
        downsample_factor_layout.addWidget(self.down_sample_factor_info)
        layout.addLayout(downsample_factor_layout)

    def init_charge_state_probabilities(self):
        info_text = "Set the probabilities for different charge states of peptides."
        self.charge_state_probabilities_group = CollapsibleBox("Charge State Probabilities", info_text)
        layout = self.charge_state_probabilities_group.content_layout

        # Load info icon image
        info_icon = QPixmap(str(self.script_dir / "info_icon.png")).scaled(19, 19, Qt.KeepAspectRatio)

        # Probability of Charge
        p_charge_layout = QHBoxLayout()
        self.p_charge_label = QLabel("Probability of Charge:")
        self.p_charge_spin = QDoubleSpinBox()
        self.p_charge_spin.setRange(0, 1)
        self.p_charge_spin.setDecimals(2)
        self.p_charge_spin.setValue(0.5)
        p_charge_layout.addWidget(self.p_charge_label)
        p_charge_layout.addWidget(self.p_charge_spin)
        self.p_charge_info = QLabel()
        self.p_charge_info.setPixmap(info_icon)
        self.p_charge_info.setToolTip("Defines the probability for a peptide to adopt a specific charge state.")
        p_charge_layout.addWidget(self.p_charge_info)
        layout.addLayout(p_charge_layout)

        # Minimum Charge Contribution
        min_charge_contrib_layout = QHBoxLayout()
        self.min_charge_contrib_label = QLabel("Minimum Charge Contribution:")
        self.min_charge_contrib_spin = QDoubleSpinBox()
        self.min_charge_contrib_spin.setRange(0, 1)
        self.min_charge_contrib_spin.setDecimals(2)
        self.min_charge_contrib_spin.setValue(0.25)
        min_charge_contrib_layout.addWidget(self.min_charge_contrib_label)
        min_charge_contrib_layout.addWidget(self.min_charge_contrib_spin)
        self.min_charge_contrib_info = QLabel()
        self.min_charge_contrib_info.setPixmap(info_icon)
        self.min_charge_contrib_info.setToolTip(
            "Specifies the minimum contribution of peptides with this charge state in the simulation."
        )
        min_charge_contrib_layout.addWidget(self.min_charge_contrib_info)
        layout.addLayout(min_charge_contrib_layout)

    def init_performance_settings(self):
        info_text = "Adjust performance-related settings, such as the number of threads and batch size."
        self.performance_settings_group = CollapsibleBox("Performance Settings", info_text)
        layout = self.performance_settings_group.content_layout

        # Load info icon image
        info_icon = QPixmap(str(self.script_dir / "info_icon.png")).scaled(19, 19, Qt.KeepAspectRatio)

        # Number of Threads
        num_threads_layout = QHBoxLayout()
        self.num_threads_label = QLabel("Number of Threads:")
        self.num_threads_spin = QSpinBox()
        self.num_threads_spin.setRange(-1, 64)
        self.num_threads_spin.setValue(-1)
        num_threads_layout.addWidget(self.num_threads_label)
        num_threads_layout.addWidget(self.num_threads_spin)

        self.num_threads_info = QLabel()
        self.num_threads_info.setPixmap(info_icon)
        self.num_threads_info.setToolTip(
            "Set the number of threads for parallel processing. "
            "Use -1 to auto-detect the number of available cores."
        )
        num_threads_layout.addWidget(self.num_threads_info)
        layout.addLayout(num_threads_layout)

        # Batch Size
        batch_size_layout = QHBoxLayout()
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10000)
        self.batch_size_spin.setValue(256)
        batch_size_layout.addWidget(self.batch_size_label)
        batch_size_layout.addWidget(self.batch_size_spin)

        self.batch_size_info = QLabel()
        self.batch_size_info.setPixmap(info_icon)
        self.batch_size_info.setToolTip(
            "Specify the batch size for processing. Larger batches may improve performance "
            "at the cost of memory usage."
        )
        batch_size_layout.addWidget(self.batch_size_info)
        layout.addLayout(batch_size_layout)

        # Add a big RUN button
        self.run_button = QPushButton("RUN Simulation")
        self.run_button.setFixedHeight(50)
        self.run_button.setFont(QFont('Arial', 14))
        # Set button style
        self.run_button.setStyleSheet("background-color: #6a994e; color: white; border: none;")
        self.run_button.clicked.connect(self.run_simulation)

        # CANCEL button (initially hidden)
        self.cancel_button = QPushButton("CANCEL Simulation")
        self.cancel_button.setFixedHeight(50)
        self.cancel_button.setFont(QFont('Arial', 14))
        self.cancel_button.setStyleSheet("background-color: #bc4749; color: white; border: none;")
        self.cancel_button.clicked.connect(self.cancel_simulation)
        self.cancel_button.setVisible(False)

        # Placeholder methods for browsing files

    def browse_save_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if directory:
            self.path_input.setText(directory)

    def browse_reference_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Reference Dataset")
        if directory:
            self.reference_input.setText(directory)

    def browse_fasta_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select FASTA File")
        if file_path:
            self.fasta_input.setText(file_path)

    def browse_existing_save_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Existing Save Directory")
        if directory:
            self.existing_path_input.setText(directory)

    def browse_multi_fasta_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Folder with Multi-FASTA Files")
        if directory:
            self.multi_fasta_input.setText(directory)

    def browse_multi_fasta_dilution_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Multi-FASTA Dilution File")
        if file_path:
            self.multi_fasta_dilution_input.setText(file_path)

    def init_console(self):
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(500)  # Adjust the height as needed
        self.console.setFont(QFont('Courier', 10))  # Use a monospace font for console output

    def run_simulation(self):
        # Collect all parameters and run the simulation
        # This function will map GUI inputs to command-line arguments

        # Collect parameters from Main Settings
        save_path = self.path_input.text()
        reference_path = self.reference_input.text()
        fasta_path = self.fasta_input.text()
        experiment_name = self.name_input.text()
        acquisition_type = self.acquisition_combo.currentText()
        use_reference_layout = self.use_reference_layout_checkbox.isChecked()
        reference_in_memory = self.reference_in_memory_checkbox.isChecked()
        sample_peptides = self.sample_peptides_checkbox.isChecked()
        sample_seed = self.sample_seed_spin.value()
        add_decoys = self.add_decoys_checkbox.isChecked()
        proteome_mix = self.proteome_mix_checkbox.isChecked()
        silent_mode = self.silent_checkbox.isChecked()
        apply_fragmentation = self.apply_fragmentation_checkbox.isChecked()

        # Collect other parameters
        num_sample_peptides = self.num_sample_peptides_spin.value()
        missed_cleavages = self.missed_cleavages_spin.value()
        min_len = self.min_len_spin.value()
        max_len = self.max_len_spin.value()
        cleave_at = self.cleave_at_input.text()
        restrict = self.restrict_input.text()
        modifications = self.mods_input.text()

        intensity_mean = 10 ** self.intensity_mean_slider.value()
        intensity_min = 10 ** self.intensity_min_slider.value()
        intensity_max = 10 ** self.intensity_max_slider.value()
        sample_occurrences = self.sample_occurrences_checkbox.isChecked()
        intensity_value = 10 ** self.intensity_value_slider.value()

        # Validation conditions
        if not (intensity_min < intensity_mean < intensity_max):
            QMessageBox.warning(
                self,
                "Invalid Intensity Settings",
                "Ensure that:\n- Minimum Intensity < Mean Intensity < Maximum Intensity.\nPlease adjust the values and try again."
            )
            return

        isotope_k = self.isotope_k_spin.value()
        isotope_min_intensity = self.isotope_min_intensity_spin.value()
        isotope_centroid = self.isotope_centroid_checkbox.isChecked()

        gradient_length = self.gradient_length_spin.value()
        mean_std_rt = self.mean_std_rt_spin.value()
        variance_std_rt = self.variance_std_rt_spin.value()
        mean_skewness = self.mean_skewness_spin.value()
        variance_skewness = self.variance_skewness_spin.value()
        std_im = self.std_im_spin.value()
        variance_std_im = self.variance_std_im_spin.value()
        z_score = self.z_score_spin.value()
        target_p = self.target_p_spin.value()
        sampling_step_size = self.sampling_step_size_spin.value()

        add_noise_to_signals = self.add_noise_to_signals_checkbox.isChecked()
        mz_noise_precursor = self.mz_noise_precursor_checkbox.isChecked()
        precursor_noise_ppm = self.precursor_noise_ppm_spin.value()
        mz_noise_fragment = self.mz_noise_fragment_checkbox.isChecked()
        fragment_noise_ppm = self.fragment_noise_ppm_spin.value()
        mz_noise_uniform = self.mz_noise_uniform_checkbox.isChecked()
        add_real_data_noise = self.add_real_data_noise_checkbox.isChecked()
        reference_noise_intensity_max = self.reference_noise_intensity_max_spin.value()
        down_sample_factor = self.down_sample_factor_spin.value()

        p_charge = self.p_charge_spin.value()
        min_charge_contrib = self.min_charge_contrib_spin.value()

        num_threads = self.num_threads_spin.value()
        batch_size = self.batch_size_spin.value()

        use_existing = self.from_existing_checkbox.isChecked()
        use_existing_path = self.existing_path_input.text()

        multi_fasta_dilution_path = None

        # Check if proteome mix is enabled
        if proteome_mix:
            fasta_path = self.multi_fasta_input.text()
            multi_fasta_dilution_path = self.multi_fasta_dilution_input.text()

        # Build the argument list
        args = [
            "timsim",
            save_path,
            reference_path,
            fasta_path,
            "--experiment_name", experiment_name,
            "--acquisition_type", acquisition_type,
            "--num_sample_peptides", str(num_sample_peptides),
            "--sample_seed", str(sample_seed),
            "--missed_cleavages", str(missed_cleavages),
            "--min_len", str(min_len),
            "--max_len", str(max_len),
            "--cleave_at", cleave_at,
            "--restrict", restrict,
            "--intensity_mean", str(intensity_mean),
            "--intensity_min", str(intensity_min),
            "--intensity_max", str(intensity_max),
            "--intensity_value", str(intensity_value),
            "--isotope_k", str(isotope_k),
            "--isotope_min_intensity", str(isotope_min_intensity),
            "--gradient_length", str(gradient_length),
            "--mean_std_rt", str(mean_std_rt),
            "--variance_std_rt", str(variance_std_rt),
            "--mean_skewness", str(mean_skewness),
            "--variance_skewness", str(variance_skewness),
            "--std_im", str(std_im),
            "--variance_std_im", str(variance_std_im),
            "--z_score", str(z_score),
            "--target_p", str(target_p),
            "--sampling_step_size", str(sampling_step_size),
            "--precursor_noise_ppm", str(precursor_noise_ppm),
            "--fragment_noise_ppm", str(fragment_noise_ppm),
            "--reference_noise_intensity_max", str(reference_noise_intensity_max),
            "--down_sample_factor", str(down_sample_factor),
            "--p_charge", str(p_charge),
            "--min_charge_contrib", str(min_charge_contrib),
            "--num_threads", str(num_threads),
            "--batch_size", str(batch_size),
            "--existing_path", str(use_existing_path),
        ]

        # Check for dilution path
        if multi_fasta_dilution_path and proteome_mix:
            args.extend(["--multi_fasta_dilution", multi_fasta_dilution_path])

        # Add boolean flags
        if not use_reference_layout:
            args.append("--no_reference_layout")
        if reference_in_memory:
            args.append("--reference_in_memory")
        if not sample_peptides:
            args.append("--no_peptide_sampling")
        if add_decoys:
            args.append("--decoys")
        if proteome_mix:
            args.append("--proteome_mix")
        if silent_mode:
            args.append("--silent_mode")
        if not sample_occurrences:
            args.append("--no_sample_occurrences")
        if not isotope_centroid:
            args.append("--no_isotope_centroid")
        if add_noise_to_signals:
            args.append("--add_noise_to_signals")
        if mz_noise_precursor:
            args.append("--mz_noise_precursor")
        if mz_noise_fragment:
            args.append("--mz_noise_fragment")
        if mz_noise_uniform:
            args.append("--mz_noise_uniform")
        if add_real_data_noise:
            args.append("--add_real_data_noise")
        if use_existing:
            args.append("--from_existing")
        if apply_fragmentation:
            args.append("--apply_fragmentation")

        # Include modifications if provided
        if modifications:
            args.extend(["--modifications", modifications])

        # Convert the list to strings
        args = [str(arg) for arg in args]

        # Ensure no existing process is running before starting a new one
        if self.process and self.process.state() == QProcess.Running:
            self.process.kill()

        # Update buttons
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("background-color: #6d6875; color: white; border: none;")
        self.cancel_button.setVisible(True)

        # Initialize QProcess
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        # Connect signals to handle output and completion
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)

        # Start the process
        self.process.start(args[0], args[1:])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode('utf-8')
        self.console.append(stdout)

    def process_finished(self):
        self.console.append("Simulation finished.")
        self.run_button.setEnabled(True)
        self.run_button.setStyleSheet("background-color: #6a994e; color: white; border: none;")

        self.cancel_button.setVisible(False)

    def cancel_simulation(self):
        if self.process is not None and self.process.state() == QProcess.Running:
            # Prompt the user to confirm cancellation
            reply = QMessageBox.question(
                self,
                "Cancel Simulation",
                "Are you sure you want to cancel the simulation?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            # If the user confirms, cancel the simulation
            if reply == QMessageBox.Yes:
                self.process.kill()
                self.console.append("Simulation canceled by user.")
                self.process_finished()  # Clean up UI and reset buttons
            else:
                self.console.append("Cancellation aborted by user.")

    def closeEvent(self, event):
        if self.process is not None and self.process.state() == QProcess.Running:
            # Prompt the user to confirm they want to terminate the running process
            reply = QMessageBox.question(
                self,
                "Exit Application",
                "A simulation is currently running. Do you want to terminate it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Kill the process and accept the event to close the application
                self.process.kill()
                event.accept()
            else:
                # Ignore the close event if the user chooses not to terminate the process
                event.ignore()
        else:
            # No running process, so close the application normally
            event.accept()

    def save_config(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config File",
            "",
            "TOML Files (*.toml);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                config = self.collect_settings()
                with open(file_path, 'w') as config_file:
                    toml.dump(config, config_file)
                QMessageBox.information(self, "Success", "Configuration saved successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save configuration:\n{e}")

    def load_config(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Config File",
            "",
            "TOML Files (*.toml);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                with open(file_path, 'r') as config_file:
                    config = toml.load(config_file)
                self.apply_settings(config)
                QMessageBox.information(self, "Success", "Configuration loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load configuration:\n{e}")

    def load_modifications(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Modifications File",
            "",
            "TOML Files (*.toml);;All Files (*)",
            options=options
        )
        if file_path:
            QMessageBox.information(self, "Success", "Modifications loaded successfully.")
            self.mods_input.setText(file_path)
        else:
            QMessageBox.warning(self, "Error", f"Failed to load modifications file.")

    def collect_settings(self):
        config = {}

        # Main Settings
        config['main_settings'] = {
            'save_path': self.path_input.text(),
            'reference_path': self.reference_input.text(),
            'fasta_path': self.fasta_input.text(),
            'experiment_name': self.name_input.text(),
            'acquisition_type': self.acquisition_combo.currentText(),
            'use_reference_layout': self.use_reference_layout_checkbox.isChecked(),
            'reference_in_memory': self.reference_in_memory_checkbox.isChecked(),
            'sample_peptides': self.sample_peptides_checkbox.isChecked(),
            'sample_seed': self.sample_seed_spin.value(),
            'add_decoys': self.add_decoys_checkbox.isChecked(),
            'proteome_mix': self.proteome_mix_checkbox.isChecked(),
            'silent_mode': self.silent_checkbox.isChecked(),
            'from_existing': self.from_existing_checkbox.isChecked(),
            'existing_path': self.existing_path_input.text(),
            'apply_fragmentation': self.apply_fragmentation_checkbox.isChecked(),
            'multi_fasta': self.multi_fasta_input.text(),
            'multi_fasta_dilution': self.multi_fasta_dilution_input.text(),
        }

        # Peptide Digestion Settings
        config['peptide_digestion'] = {
            'num_sample_peptides': self.num_sample_peptides_spin.value(),
            'missed_cleavages': self.missed_cleavages_spin.value(),
            'min_len': self.min_len_spin.value(),
            'max_len': self.max_len_spin.value(),
            'cleave_at': self.cleave_at_input.text(),
            'restrict': self.restrict_input.text(),
            'modifications': self.mods_input.text(),
        }

        # Peptide Intensity Settings
        config['peptide_intensity'] = {
            'intensity_mean': self.intensity_mean_slider.value(),
            'intensity_min': self.intensity_min_slider.value(),
            'intensity_max': self.intensity_max_slider.value(),
            'sample_occurrences': self.sample_occurrences_checkbox.isChecked(),
            'intensity_value': self.intensity_value_slider.value(),
        }

        # Isotopic Pattern Settings
        config['isotopic_pattern'] = {
            'isotope_k': self.isotope_k_spin.value(),
            'isotope_min_intensity': self.isotope_min_intensity_spin.value(),
            'isotope_centroid': self.isotope_centroid_checkbox.isChecked(),
        }

        # Distribution Settings
        config['distribution_settings'] = {
            'gradient_length': self.gradient_length_spin.value(),
            'mean_std_rt': self.mean_std_rt_spin.value(),
            'variance_std_rt': self.variance_std_rt_spin.value(),
            'mean_skewness': self.mean_skewness_spin.value(),
            'variance_skewness': self.variance_skewness_spin.value(),
            'std_im': self.std_im_spin.value(),
            'variance_std_im': self.variance_std_im_spin.value(),
            'z_score': self.z_score_spin.value(),
            'target_p': self.target_p_spin.value(),
            'sampling_step_size': self.sampling_step_size_spin.value(),
        }

        # Noise Settings
        config['noise_settings'] = {
            'add_noise_to_signals': self.add_noise_to_signals_checkbox.isChecked(),
            'mz_noise_precursor': self.mz_noise_precursor_checkbox.isChecked(),
            'precursor_noise_ppm': self.precursor_noise_ppm_spin.value(),
            'mz_noise_fragment': self.mz_noise_fragment_checkbox.isChecked(),
            'fragment_noise_ppm': self.fragment_noise_ppm_spin.value(),
            'mz_noise_uniform': self.mz_noise_uniform_checkbox.isChecked(),
            'add_real_data_noise': self.add_real_data_noise_checkbox.isChecked(),
            'reference_noise_intensity_max': self.reference_noise_intensity_max_spin.value(),
            'down_sample_factor': self.down_sample_factor_spin.value(),
        }

        # Charge State Probabilities
        config['charge_state_probabilities'] = {
            'p_charge': self.p_charge_spin.value(),
            'min_charge_contrib': self.min_charge_contrib_spin.value(),
        }

        # Performance Settings
        config['performance_settings'] = {
            'num_threads': self.num_threads_spin.value(),
            'batch_size': self.batch_size_spin.value(),
        }

        return config

    def apply_settings(self, config):
        # Main Settings
        main_settings = config.get('main_settings', {})
        self.path_input.setText(main_settings.get('save_path', ''))
        self.reference_input.setText(main_settings.get('reference_path', ''))
        self.fasta_input.setText(main_settings.get('fasta_path', ''))
        self.name_input.setText(main_settings.get('experiment_name', f"TIMSIM-{int(time.time())}"))
        acquisition_type = main_settings.get('acquisition_type', 'DIA')
        index = self.acquisition_combo.findText(acquisition_type)
        if index >= 0:
            self.acquisition_combo.setCurrentIndex(index)
        self.use_reference_layout_checkbox.setChecked(main_settings.get('use_reference_layout', True))
        self.reference_in_memory_checkbox.setChecked(main_settings.get('reference_in_memory', False))
        self.sample_peptides_checkbox.setChecked(main_settings.get('sample_peptides', True))
        self.sample_seed_spin.setValue(main_settings.get('sample_seed', 41))
        self.add_decoys_checkbox.setChecked(main_settings.get('add_decoys', False))
        self.proteome_mix_checkbox.setChecked(main_settings.get('proteome_mix', False))
        self.silent_checkbox.setChecked(main_settings.get('silent_mode', False))
        self.apply_fragmentation_checkbox.setChecked(main_settings.get('fragmentation', True))
        self.from_existing_checkbox.setChecked(main_settings.get('from_existing', False))
        self.existing_path_input.setText(main_settings.get('existing_path', ''))
        self.multi_fasta_input.setText(main_settings.get('multi_fasta', ''))
        self.multi_fasta_dilution_input.setText(main_settings.get('multi_fasta_dilution', ''))

        # Peptide Digestion Settings
        peptide_digestion = config.get('peptide_digestion', {})
        self.num_sample_peptides_spin.setValue(peptide_digestion.get('num_sample_peptides', 25000))
        self.missed_cleavages_spin.setValue(peptide_digestion.get('missed_cleavages', 2))
        self.min_len_spin.setValue(peptide_digestion.get('min_len', 7))
        self.max_len_spin.setValue(peptide_digestion.get('max_len', 30))
        self.cleave_at_input.setText(peptide_digestion.get('cleave_at', 'KR'))
        self.restrict_input.setText(peptide_digestion.get('restrict', 'P'))
        self.mods_input.setText(peptide_digestion.get('modifications', ''))

        # Peptide Intensity Settings
        peptide_intensity = config.get('peptide_intensity', {})
        self.intensity_mean_slider.setValue(peptide_intensity.get('intensity_mean', 1e7))
        self.intensity_min_slider.setValue(peptide_intensity.get('intensity_min', 1e5))
        self.intensity_max_slider.setValue(peptide_intensity.get('intensity_max', 1e9))
        self.sample_occurrences_checkbox.setChecked(peptide_intensity.get('sample_occurrences', True))
        self.intensity_value_slider.setValue(peptide_intensity.get('intensity_value', 1e6))

        # Isotopic Pattern Settings
        isotopic_pattern = config.get('isotopic_pattern', {})
        self.isotope_k_spin.setValue(isotopic_pattern.get('isotope_k', 8))
        self.isotope_min_intensity_spin.setValue(isotopic_pattern.get('isotope_min_intensity', 1))
        self.isotope_centroid_checkbox.setChecked(isotopic_pattern.get('isotope_centroid', True))

        # Distribution Settings
        distribution_settings = config.get('distribution_settings', {})
        self.gradient_length_spin.setValue(distribution_settings.get('gradient_length', 3600))
        self.mean_std_rt_spin.setValue(distribution_settings.get('mean_std_rt', 1.5))
        self.variance_std_rt_spin.setValue(distribution_settings.get('variance_std_rt', 0.3))
        self.mean_skewness_spin.setValue(distribution_settings.get('mean_skewness', 0.3))
        self.variance_skewness_spin.setValue(distribution_settings.get('variance_skewness', 0.1))
        self.std_im_spin.setValue(distribution_settings.get('std_im', 0.01))
        self.variance_std_im_spin.setValue(distribution_settings.get('variance_std_im', 0.001))
        self.variance_std_im_spin.setValue(distribution_settings.get('variance_std_im', 0.001))
        self.z_score_spin.setValue(distribution_settings.get('z_score', 0.99))
        self.target_p_spin.setValue(distribution_settings.get('target_p', 0.999))
        self.sampling_step_size_spin.setValue(distribution_settings.get('sampling_step_size', 0.001))

        # Noise Settings
        noise_settings = config.get('noise_settings', {})
        self.add_noise_to_signals_checkbox.setChecked(noise_settings.get('add_noise_to_signals', False))
        self.mz_noise_precursor_checkbox.setChecked(noise_settings.get('mz_noise_precursor', False))
        self.precursor_noise_ppm_spin.setValue(noise_settings.get('precursor_noise_ppm', 5.0))
        self.mz_noise_fragment_checkbox.setChecked(noise_settings.get('mz_noise_fragment', False))
        self.fragment_noise_ppm_spin.setValue(noise_settings.get('fragment_noise_ppm', 5.0))
        self.mz_noise_uniform_checkbox.setChecked(noise_settings.get('mz_noise_uniform', False))
        self.add_real_data_noise_checkbox.setChecked(noise_settings.get('add_real_data_noise', False))
        self.reference_noise_intensity_max_spin.setValue(noise_settings.get('reference_noise_intensity_max', 30))
        self.down_sample_factor_spin.setValue(noise_settings.get('down_sample_factor', 0.5))

        # Charge State Probabilities
        charge_state_probabilities = config.get('charge_state_probabilities', {})
        self.p_charge_spin.setValue(charge_state_probabilities.get('p_charge', 0.5))
        self.min_charge_contrib_spin.setValue(charge_state_probabilities.get('min_charge_contrib', 0.25))

        # Performance Settings
        performance_settings = config.get('performance_settings', {})
        self.num_threads_spin.setValue(performance_settings.get('num_threads', -1))
        self.batch_size_spin.setValue(performance_settings.get('batch_size', 256))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TimSim")
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
