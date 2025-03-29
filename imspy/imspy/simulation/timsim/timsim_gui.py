import sys
import time
from pathlib import Path
import markdown
import qdarkstyle
import toml
import numpy as np
from scipy.stats import exponnorm
from imspy.simulation.timsim.simulator import calculate_rt_defaults

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, QSpinBox, QToolButton,
    QDoubleSpinBox, QComboBox, QGroupBox, QScrollArea, QAction, QMessageBox,
    QTextEdit, QTabWidget
)
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QFont, QIcon, QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


# =============================================================================
# VTK Visualizer (unchanged)
# =============================================================================
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
        self.renderer.RemoveAllViewProps()
        try:
            # Example data loading – adjust the path as needed
            from imspy.timstof import TimsDatasetDIA
            dia_data = TimsDatasetDIA("/media/hd02/data/sim/dia/timsim-gui/plain/example.d")
            t_slice = dia_data.get_tims_slice(dia_data.precursor_frames[500:1000]).filter(
                mz_min=500, mz_max=800, intensity_min=5, mobility_min=0.5, mobility_max=1.1)
            mz = t_slice.df.mz.values
            rt = t_slice.df.retention_time.values
            intensity = t_slice.df.intensity.values
            im = t_slice.df.mobility.values
            self.visualize_data(rt, im, mz, intensity)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load data:\n{e}")

    def visualize_data(self, rt, im, mz, intensity):
        rt_min, rt_max = rt.min(), rt.max()
        im_min, im_max = im.min(), im.max()
        mz_min, mz_max = mz.min(), mz.max()

        # Normalize for a cubic aspect ratio
        rt_norm = (rt - rt_min) / (rt_max - rt_min)
        im_norm = (im - im_min) / (im_max - im_min)
        mz_norm = (mz - mz_min) / (mz_max - mz_min)

        # Log scale intensity
        log_intensity = np.log10(intensity)
        cmap = plt.get_cmap("inferno")
        norm_log_intensity = (log_intensity - log_intensity.min()) / (log_intensity.max() - log_intensity.min())
        colors_inferno = (cmap(norm_log_intensity)[:, :3] * 255).astype(np.uint8)

        mapper = self.create_vtk_point_cloud(rt_norm, im_norm, mz_norm, colors_inferno)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.6)
        self.renderer.AddActor(actor)

        bounds = [0, 1, 0, 1, 0, 1]
        cube_axes_actor = self.create_cube_axes(bounds, rt_min, rt_max, im_min, im_max, mz_min, mz_max)
        self.renderer.AddActor(cube_axes_actor)

        self.renderer.SetBackground(0.9, 0.9, 0.9)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def create_vtk_point_cloud(self, rt, im, mz, colors_inferno):
        points = vtk.vtkPoints()
        vtk_points_array = numpy_to_vtk(np.column_stack((rt, im, mz)).astype(np.float32), deep=True)
        points.SetData(vtk_points_array)
        vtk_colors = numpy_to_vtk(colors_inferno, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_colors.SetName("Colors")
        point_cloud = vtk.vtkPolyData()
        point_cloud.SetPoints(points)
        point_cloud.GetPointData().SetScalars(vtk_colors)
        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetInputData(point_cloud)
        mapper.ScalarVisibilityOn()
        mapper.SetColorModeToDefault()
        glyph = vtk.vtkSphereSource()
        glyph.SetRadius(0.001)
        mapper.SetSourceConnection(glyph.GetOutputPort())
        return mapper

    def create_cube_axes(self, bounds, rt_min, rt_max, im_min, im_max, mz_min, mz_max):
        axes = vtk.vtkCubeAxesActor()
        axes.SetBounds(bounds)
        axes.SetCamera(self.renderer.GetActiveCamera())
        axes.SetXTitle("Retention Time (s)")
        axes.SetYTitle("Ion Mobility (1/K0)")
        axes.SetZTitle("m/z")
        axes.SetXUnits(f"[{rt_min:.1f} - {rt_max:.1f}]")
        axes.SetYUnits(f"[{im_min:.1f} - {im_max:.1f}]")
        axes.SetZUnits(f"[{mz_min:.1f} - {mz_max:.1f}]")
        for i in range(3):
            axes.GetTitleTextProperty(i).SetColor(0, 0, 0)
            axes.GetLabelTextProperty(i).SetColor(0, 0, 0)
        axes.DrawXGridlinesOn()
        axes.DrawYGridlinesOn()
        axes.DrawZGridlinesOn()
        axes.SetGridLineLocation(axes.VTK_GRID_LINES_FURTHEST)
        return axes


# =============================================================================
# Collapsible Box for Grouping Settings
# =============================================================================
class CollapsibleBox(QWidget):
    def __init__(self, title="", info_text="", parent=None):
        super().__init__(parent)
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)
        self.info_label = QLabel()
        icon_path = Path(__file__).parent.parent / "resources/icons/info_icon.png"
        if icon_path.exists():
            self.info_label.setPixmap(QPixmap(str(icon_path)).scaled(19, 19, Qt.KeepAspectRatio))
            self.info_label.setToolTip(info_text)
        self.header_layout = QHBoxLayout()
        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.info_label)
        self.header_layout.addStretch()
        self.content_area = QWidget()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(25, 0, 0, 0)
        self.content_area.setLayout(self.content_layout)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(self.header_layout)
        main_layout.addWidget(self.content_area)
        self.setLayout(main_layout)

    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        if checked:
            self.content_area.setMaximumHeight(16777215)
            self.content_area.setMinimumHeight(0)
        else:
            self.content_area.setMaximumHeight(0)
            self.content_area.setMinimumHeight(0)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


# =============================================================================
# EMGPlot for Distribution Visualization (unchanged)
# =============================================================================
class EMGPlot(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 2)
        super().__init__(self.fig)
        self.mu = 0
        self.central_interval_p = 0.76
        # TODO redefinition of defaults is bad practice

        default_param_sigma_lower_upper = calculate_rt_defaults(3600)
        self.sigma_lower = default_param_sigma_lower_upper["sigma_lower_rt"]
        self.sigma_upper = default_param_sigma_lower_upper["sigma_upper_rt"]
        self.sigma_alpha = 4
        self.sigma_beta = 4
        self.k_lower = 0
        self.k_upper = 10
        self.k_alpha = 1
        self.k_beta = 20
        self.N = 200
        self.x = np.linspace(-20, 20, 1000)
        self.update_plot()

    def emg_pdf(self, x, mu, sigma, k):
        return exponnorm.pdf(x, K=k, loc=mu, scale=sigma)

    def central_interval(self, mu, sigma, k):
        p = self.central_interval_p
        higher = exponnorm.ppf(1 - (1 - p) / 2, K=k, loc=mu, scale=sigma)
        lower = exponnorm.ppf((1 - p) / 2, K=k, loc=mu, scale=sigma)
        return higher - lower

    def update_plot(self):
        for ax in self.ax:
            ax.clear()
        self.ax[0].set_xlim(-10, 10)
        self.ax[1].set_xlim(0, 20)
        central_intervals = []
        for _ in range(self.N):
            # TODO use rng
            sampled_sigma = np.random.beta(a=self.sigma_alpha, b=self.sigma_beta) * (
                        self.sigma_upper - self.sigma_lower) + self.sigma_lower
            sampled_k = np.random.beta(a=self.k_alpha, b=self.k_beta) * (self.k_upper - self.k_lower) + self.k_lower
            y = self.emg_pdf(self.x, self.mu, sampled_sigma, sampled_k)
            central_intervals.append(self.central_interval(self.mu, sampled_sigma, sampled_k))
            self.ax[0].plot(self.x, y, alpha=0.5)

        self.ax[0].set_title("EMG Distributions")
        self.ax[0].set_xlabel("Retention Time [Seconds]")
        self.ax[0].set_ylabel("Intensity")
        self.ax[1].hist(central_intervals, bins=20)
        ci_mean = np.mean(central_intervals)
        self.ax[1].vlines(ci_mean, 0, self.ax[1].get_ylim()[1] * 0.9, color='r', linestyle='--')
        self.ax[1].text(ci_mean, self.ax[1].get_ylim()[1] * 0.91, f"Mean: {ci_mean:.2f}s")
        self.ax[1].set_title(f"Central Interval ({self.central_interval_p * 100:.1f}%) Sizes")
        self.ax[1].set_xlabel(f"Central Interval ({self.central_interval_p * 100:.1f}%) [Seconds]")
        self.ax[1].set_ylabel("Count")
        self.draw()

    def set_sigma_lower(self, value):
        self.sigma_lower = value
        self.update_plot()

    def set_sigma_upper(self, value):
        self.sigma_upper = value
        self.update_plot()

    def set_sigma_alpha(self, value):
        self.sigma_alpha = value
        self.update_plot()

    def set_sigma_beta(self, value):
        self.sigma_beta = value
        self.update_plot()

    def set_k_lower(self, value):
        self.k_lower = value
        self.update_plot()

    def set_k_upper(self, value):
        self.k_upper = value
        self.update_plot()

    def set_k_alpha(self, value):
        self.k_alpha = value
        self.update_plot()

    def set_k_beta(self, value):
        self.k_beta = value
        self.update_plot()

    # =============================================================================


# MainWindow – The Main GUI
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.script_dir = Path(__file__).parent.parent / "resources/icons/"
        self.setWindowTitle("🦀💻 TimSim 🔬🐍 - Proteomics Experiment Simulation on timsTOF")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon(str(self.script_dir / "logo_2.png")))
        self.info_icon = QPixmap(str(self.script_dir / "info_icon.png")).scaled(19, 19, Qt.KeepAspectRatio)
        self.init_menu_bar()
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        self.create_main_tab()
        self.create_documentation_tab()
        self.process = None

    def create_documentation_tab(self):
        documentation_tab = QWidget()
        layout = QVBoxLayout(documentation_tab)
        doc_path = Path(__file__).parent.parent / "resources/docs/documentation.md"
        if doc_path.exists():
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    markdown_text = f.read()
                html_content = markdown.markdown(markdown_text)
            except Exception as e:
                html_content = f"<p>Error loading documentation: {str(e)}</p>"
        else:
            html_content = "<p>Documentation file not found.</p>"
        self.documentation_viewer = QTextEdit()
        self.documentation_viewer.setHtml(html_content)
        self.documentation_viewer.setReadOnly(True)
        layout.addWidget(self.documentation_viewer)
        self.tab_widget.addTab(documentation_tab, "Documentation")

    def create_main_tab(self):
        main_tab = QWidget()
        self.main_layout = QVBoxLayout(main_tab)
        self.init_main_settings()
        self.init_peptide_digestion_settings()
        self.init_isotopic_pattern_settings()
        self.init_distribution_settings()
        self.init_noise_settings()
        self.init_charge_state_probabilities()
        self.init_performance_settings()
        self.init_dda_settings()
        self.init_console()
        self.main_layout.addWidget(self.main_settings_group)
        self.main_layout.addWidget(self.peptide_digestion_group)
        self.main_layout.addWidget(self.isotopic_pattern_group)
        self.main_layout.addWidget(self.distribution_settings_group)
        self.main_layout.addWidget(self.noise_settings_group)
        self.main_layout.addWidget(self.charge_state_probabilities_group)
        self.main_layout.addWidget(self.performance_settings_group)
        self.main_layout.addWidget(self.dda_settings_group)
        self.main_layout.addWidget(self.console)
        self.main_layout.addWidget(self.run_button)
        self.main_layout.addWidget(self.cancel_button)
        self.main_layout.addStretch()
        self.logo_label = QLabel()
        pixmap = QPixmap(str(self.script_dir / "rustims_logo.png"))
        scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(scaled_pixmap)
        self.logo_label.setFixedSize(200, 200)
        logo_layout = QHBoxLayout()
        logo_layout.addStretch()
        logo_layout.addWidget(self.logo_label)
        logo_layout.addStretch()
        self.main_layout.addLayout(logo_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(main_tab)
        self.tab_widget.addTab(scroll, "Main Settings")

    def init_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        save_action = QAction('Save Config', self)
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
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
        return setting_widget

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
        setattr(self, attribute_name, spinbox)

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
        self.add_setting_with_info(layout, "Save Path:", path_widget,
                                   "Select the directory to save simulation outputs.")
        # Reference Dataset Path
        self.reference_input = QLineEdit()
        self.reference_browse = QPushButton("Browse")
        self.reference_browse.clicked.connect(self.browse_reference_path)
        reference_widget = QWidget()
        reference_layout = QHBoxLayout(reference_widget)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.addWidget(self.reference_input)
        reference_layout.addWidget(self.reference_browse)
        self.add_setting_with_info(layout, "Reference Dataset Path:", reference_widget,
                                   "Specify the path to the reference dataset used as a template for simulation.")
        # FASTA File Path
        self.fasta_input = QLineEdit()
        self.fasta_browse = QPushButton("Browse")
        self.fasta_browse.clicked.connect(self.browse_fasta_path)
        fasta_widget = QWidget()
        fasta_layout = QHBoxLayout(fasta_widget)
        fasta_layout.setContentsMargins(0, 0, 0, 0)
        fasta_layout.addWidget(self.fasta_input)
        fasta_layout.addWidget(self.fasta_browse)
        self.add_setting_with_info(layout, "FASTA File Path:", fasta_widget,
                                   "Provide the path to the FASTA file containing protein sequences for digestion.")
        # Experiment Name
        self.name_input = QLineEdit(f"TIMSIM-{int(time.time())}")
        self.add_setting_with_info(layout, "Experiment Name:", self.name_input,
                                   "Name your experiment to identify it in saved outputs and reports.")
        # Acquisition Type
        self.acquisition_combo = QComboBox()
        self.acquisition_combo.addItems(["DIA", "DDA", "SYNCHRO", "SLICE", "MIDIA"])
        self.add_setting_with_info(layout, "Acquisition Type:", self.acquisition_combo,
                                   "Select the acquisition method used to simulate data.")
        # Options (checkboxes)
        options_layout = QVBoxLayout()
        self.use_reference_layout_checkbox = QCheckBox("Use Reference Layout")
        self.use_reference_layout_checkbox.setChecked(True)
        self.add_checkbox_with_info(options_layout, self.use_reference_layout_checkbox,
                                    "Use the layout of the reference dataset as a template for simulation.")
        self.reference_in_memory_checkbox = QCheckBox("Load Reference into Memory")
        self.reference_in_memory_checkbox.setChecked(False)
        self.add_checkbox_with_info(options_layout, self.reference_in_memory_checkbox,
                                    "Load the reference dataset entirely into memory to speed up access.")
        self.sample_peptides_checkbox = QCheckBox("Sample Peptides")
        self.sample_peptides_checkbox.setChecked(True)
        self.add_checkbox_with_info(options_layout, self.sample_peptides_checkbox,
                                    "Enable to randomly sample peptides for a subset of the dataset.")
        self.sample_seed_spin = QSpinBox()
        self.sample_seed_spin.setRange(0, 1000000)
        self.sample_seed_spin.setValue(41)
        self.add_setting_with_info(options_layout, "Sample Seed:", self.sample_seed_spin,
                                   "Set the seed for the random peptide sampling.")
        self.add_decoys_checkbox = QCheckBox("Generate Decoys")
        self.add_decoys_checkbox.setChecked(False)
        self.add_checkbox_with_info(options_layout, self.add_decoys_checkbox,
                                    "Also simulate inverted decoy peptides (non-biological).")
        self.silent_checkbox = QCheckBox("Silent Mode")
        self.silent_checkbox.setChecked(False)
        self.add_checkbox_with_info(options_layout, self.silent_checkbox,
                                    "Run the simulation in silent mode, suppressing most output messages.")
        # Note: Changed default to unchecked to match new CLI tool
        self.apply_fragmentation_checkbox = QCheckBox("Apply Fragmentation to Ions")
        self.apply_fragmentation_checkbox.setChecked(True)
        self.add_checkbox_with_info(options_layout, self.apply_fragmentation_checkbox,
                                    "Enable to perform fragmentation on the ions. (Default is off)")
        self.from_existing_checkbox = QCheckBox("Use Existing Simulation Data as Template")
        self.from_existing_checkbox.setChecked(False)
        self.add_checkbox_with_info(options_layout, self.from_existing_checkbox,
                                    "Use existing simulated data as a template for the current simulation.")
        self.proteome_mix_checkbox = QCheckBox("Proteome Mixture")
        self.proteome_mix_checkbox.setChecked(False)
        self.add_checkbox_with_info(options_layout, self.proteome_mix_checkbox,
                                    "Simulate a mixture of proteomes for a more complex sample.")
        # NEW: Phospho Mode Checkbox
        self.phospho_mode_checkbox = QCheckBox("Phospho Mode")
        self.phospho_mode_checkbox.setChecked(False)
        self.add_checkbox_with_info(options_layout, self.phospho_mode_checkbox,
                                    "Enable phospho mode to generate a phospho-enriched dataset for testing localization algorithms.")
        layout.addLayout(options_layout)
        # Existing simulation path (visible if from_existing is checked)
        self.existing_path_input = QLineEdit()
        self.existing_path_browse = QPushButton("Browse")
        self.existing_path_browse.clicked.connect(self.browse_existing_save_path)
        existing_path_widget = QWidget()
        existing_path_layout = QHBoxLayout(existing_path_widget)
        existing_path_layout.setContentsMargins(0, 0, 0, 0)
        existing_path_layout.addWidget(self.existing_path_input)
        existing_path_layout.addWidget(self.existing_path_browse)
        existing_setting_widget = self.add_setting_with_info(layout, "Existing Simulation Path:", existing_path_widget,
                                                             "Select the directory with existing simulated data.")
        existing_setting_widget.setVisible(False)
        self.from_existing_checkbox.toggled.connect(existing_setting_widget.setVisible)
        # Multi-FASTA options (visible if proteome_mix is checked)
        self.multi_fasta_input = QLineEdit()
        self.multi_fasta_browse = QPushButton("Browse")
        self.multi_fasta_browse.clicked.connect(self.browse_multi_fasta_path)
        multi_fasta_widget = QWidget()
        multi_fasta_layout = QHBoxLayout(multi_fasta_widget)
        multi_fasta_layout.setContentsMargins(0, 0, 0, 0)
        multi_fasta_layout.addWidget(self.multi_fasta_input)
        multi_fasta_layout.addWidget(self.multi_fasta_browse)
        multi_fasta_setting_widget = self.add_setting_with_info(layout, "Multi-FASTA Path:", multi_fasta_widget,
                                                                "Select the folder containing multiple FASTA files for the proteome mixture.")
        multi_fasta_setting_widget.setVisible(False)
        self.multi_fasta_dilution_input = QLineEdit()
        self.multi_fasta_dilution_browse = QPushButton("Browse")
        self.multi_fasta_dilution_browse.clicked.connect(self.browse_multi_fasta_dilution_path)
        multi_fasta_dilution_widget = QWidget()
        multi_fasta_dilution_layout = QHBoxLayout(multi_fasta_dilution_widget)
        multi_fasta_dilution_layout.setContentsMargins(0, 0, 0, 0)
        multi_fasta_dilution_layout.addWidget(self.multi_fasta_dilution_input)
        multi_fasta_dilution_layout.addWidget(self.multi_fasta_dilution_browse)
        multi_fasta_dilution_setting_widget = self.add_setting_with_info(layout, "Multi-FASTA Dilution File:",
                                                                         multi_fasta_dilution_widget,
                                                                         "Select the CSV file with dilution factors for the proteome mixture.")
        multi_fasta_dilution_setting_widget.setVisible(False)
        self.proteome_mix_checkbox.toggled.connect(multi_fasta_setting_widget.setVisible)
        self.proteome_mix_checkbox.toggled.connect(multi_fasta_dilution_setting_widget.setVisible)
        self.main_settings_group.setLayout(layout)
        self.main_layout.addWidget(self.main_settings_group)

    def init_peptide_digestion_settings(self):
        info_text = "Configure peptide generation from protein sequences, including cleavage rules and peptide length constraints."
        self.peptide_digestion_group = CollapsibleBox("Peptide Digestion Settings", info_text)
        layout = self.peptide_digestion_group.content_layout
        self.num_sample_peptides_spin = QSpinBox()

        self.sample_occurrences_checkbox = QCheckBox("Sample Occurrences Randomly")
        self.sample_occurrences_checkbox.setChecked(True)
        layout.addWidget(self.sample_occurrences_checkbox)
        occ_info = QLabel()
        occ_info.setPixmap(self.info_icon.scaled(19, 19))
        occ_info.setToolTip("If enabled, peptide occurrences will be randomly sampled.")
        layout.addWidget(occ_info)

        self.num_sample_peptides_spin.setRange(1, 1000000)
        self.num_sample_peptides_spin.setValue(25000)
        self.add_setting_with_info(layout, "Number of Sampled Peptides:", self.num_sample_peptides_spin,
                                   "Total number of peptides to sample from the digested FASTA.")
        self.missed_cleavages_spin = QSpinBox()
        self.missed_cleavages_spin.setRange(0, 10)
        self.missed_cleavages_spin.setValue(2)
        self.add_setting_with_info(layout, "Missed Cleavages:", self.missed_cleavages_spin,
                                   "Number of allowed missed cleavages during digestion.")
        self.min_len_spin = QSpinBox()
        self.min_len_spin.setRange(1, 100)
        self.min_len_spin.setValue(7)
        self.add_setting_with_info(layout, "Minimum Peptide Length:", self.min_len_spin,
                                   "Minimum length (in amino acids) for peptides.")
        self.max_len_spin = QSpinBox()
        self.max_len_spin.setRange(1, 100)
        self.max_len_spin.setValue(30)
        self.add_setting_with_info(layout, "Maximum Peptide Length:", self.max_len_spin,
                                   "Maximum length (in amino acids) for peptides.")
        self.cleave_at_input = QLineEdit("KR")
        self.add_setting_with_info(layout, "Cleave At:", self.cleave_at_input,
                                   "Residues at which cleavage occurs (e.g., 'KR').")
        self.restrict_input = QLineEdit("P")
        self.add_setting_with_info(layout, "Restrict:", self.restrict_input,
                                   "Residues that restrict cleavage (e.g., 'P').")
        self.mods_input = QLineEdit()
        self.mods_browse = QPushButton("Browse")
        self.mods_browse.clicked.connect(self.load_modifications)
        mods_widget = QWidget()
        mods_layout = QHBoxLayout(mods_widget)
        mods_layout.setContentsMargins(0, 0, 0, 0)
        mods_layout.addWidget(self.mods_input)
        mods_layout.addWidget(self.mods_browse)
        self.add_setting_with_info(layout, "Amino Acid Modifications:", mods_widget,
                                   "Path to the modifications TOML file (fixed and variable modifications).")
        self.main_layout.addWidget(self.peptide_digestion_group)

    def init_dda_settings(self):
        info_text = "Configure DDA selection parameters."
        self.dda_settings_group = CollapsibleBox("DDA Settings", info_text)
        layout = self.dda_settings_group.content_layout

        # Precursors Every
        self.precursors_every_spin = QSpinBox()
        self.precursors_every_spin.setRange(2, 50)
        self.precursors_every_spin.setValue(7)
        self.add_setting_with_info(layout, "Precursors Every:", self.precursors_every_spin,
                                   "Number of precursors to select every cycle (default: 10)")

        # Precursor Intensity Threshold
        self.precursor_intensity_threshold_spin = QDoubleSpinBox()
        self.precursor_intensity_threshold_spin.setRange(0, 1e6)
        self.precursor_intensity_threshold_spin.setValue(500)
        self.add_setting_with_info(layout, "Precursor Intensity Threshold:", self.precursor_intensity_threshold_spin,
                                   "Intensity threshold for precursor selection (default: 500)")

        # Maximum Precursors
        self.max_precursors_spin = QSpinBox()
        self.max_precursors_spin.setRange(1, 1000)
        self.max_precursors_spin.setValue(7)
        self.add_setting_with_info(layout, "Max Precursors:", self.max_precursors_spin,
                                   "Maximum number of precursors to select per cycle (default: 25)")

        # Exclusion Width
        self.exclusion_width_spin = QSpinBox()
        self.exclusion_width_spin.setRange(1, 1000)
        self.exclusion_width_spin.setValue(25)
        self.add_setting_with_info(layout, "Exclusion Width:", self.exclusion_width_spin,
                                   "Exclusion width for precursor selection (default: 25)")

        # Selection Mode using QComboBox with fixed options:
        self.selection_mode_combo = QComboBox()
        self.selection_mode_combo.addItems(["topN", "random"])
        self.add_setting_with_info(layout, "Selection Mode:", self.selection_mode_combo,
                                   "Selection mode for precursors (default: topN)")

        # Add the group to your main layout
        self.main_layout.addWidget(self.dda_settings_group)

    def init_isotopic_pattern_settings(self):
        info_text = "Configure isotopic pattern generation (number of isotopes, intensity thresholds, centroiding)."
        self.isotopic_pattern_group = CollapsibleBox("Isotopic Pattern Settings", info_text)
        layout = self.isotopic_pattern_group.content_layout
        isotope_k_label = QLabel("Maximum number of Isotopes:")
        self.isotope_k_spin = QSpinBox()
        self.isotope_k_spin.setRange(1, 20)
        self.isotope_k_spin.setValue(8)
        layout.addWidget(isotope_k_label)
        layout.addWidget(self.isotope_k_spin)
        k_info = QLabel()
        k_info.setPixmap(self.info_icon)
        k_info.setToolTip("Maximum isotopes to simulate for each peptide.")
        layout.addWidget(k_info)
        isotope_min_intensity_label = QLabel("Minimum Isotope Intensity:")
        self.isotope_min_intensity_spin = QSpinBox()
        self.isotope_min_intensity_spin.setRange(0, 1000)
        self.isotope_min_intensity_spin.setValue(1)
        layout.addWidget(isotope_min_intensity_label)
        layout.addWidget(self.isotope_min_intensity_spin)
        min_info = QLabel()
        min_info.setPixmap(self.info_icon)
        min_info.setToolTip("Threshold intensity for the smallest isotope.")
        layout.addWidget(min_info)
        self.isotope_centroid_checkbox = QCheckBox("Centroid Isotopes")
        self.isotope_centroid_checkbox.setChecked(True)
        layout.addWidget(self.isotope_centroid_checkbox)
        centroid_info = QLabel()
        centroid_info.setPixmap(self.info_icon)
        centroid_info.setToolTip("Enable to centroid isotopic peaks.")
        layout.addWidget(centroid_info)
        self.main_layout.addWidget(self.isotopic_pattern_group)

    def init_distribution_settings(self):
        info_text = "Adjust signal distribution parameters including gradient length, retention time spread, and ion mobility spread."
        self.distribution_settings_group = CollapsibleBox("Signal Distribution Settings", info_text)
        main_layout = QHBoxLayout()
        settings_layout = QVBoxLayout()
        plot_layout = QVBoxLayout()
        # List of settings – note the two new ones for ion mobility
        sigma_lower_upper_start = calculate_rt_defaults(3600)
        settings = [
            {
                'attribute_name': 'gradient_length_spin',
                'label_text': 'Gradient Length (seconds):',
                'tooltip_text': 'Total length of the simulated chromatographic gradient in seconds.',
                'min_value': 1,
                'max_value': 1e5,
                'default_value': 3600,
                'decimals': 0
            },
            {
                'attribute_name': 'sigma_lower_rt_spin',
                'label_text': 'Lower sigma RT:',
                'tooltip_text': 'Lower bound for sigma of an EMG chromatographic peak (affects peak widths).',
                'min_value': 0.01,
                'max_value': 100,
                'default_value': sigma_lower_upper_start["sigma_lower_rt"],
                'decimals': 3
            },
            {
                'attribute_name': 'sigma_upper_rt_spin',
                'label_text': 'Upper sigma RT:',
                'tooltip_text': 'Upper bound for sigma of an EMG chromatographic peak (affects peak widths, must be higher than sigma_lower_rt).',
                'min_value': 0.01,
                'max_value': 100,
                'default_value': sigma_lower_upper_start["sigma_upper_rt"],
                'decimals': 3
            },
            {
                'attribute_name': 'sigma_alpha_rt_spin',
                'label_text': 'Alpha sigma RT:',
                'tooltip_text': 'Alpha for beta distribution for sigma_hat that is then scaled to sigma in (sigma_lower_rt, sigma_upper_rt).',
                'min_value': 0.01,
                'max_value': 100,
                'default_value': 4,
                'decimals': 3
            },
            {
                'attribute_name': 'sigma_beta_rt_spin',
                'label_text': 'Beta sigma RT:',
                'tooltip_text': 'Beta for beta distribution for sigma_hat that is then scaled to sigma in (sigma_lower_rt, sigma_upper_rt).',
                'min_value': 0.01,
                'max_value': 100,
                'default_value': 4,
                'decimals': 3
            },
            {
                'attribute_name': 'k_lower_rt_spin',
                'label_text': 'Lower k RT:',
                'tooltip_text': 'Lower bound for k of an EMG chromatographic peak (affects peak skewness).',
                'min_value': 0,
                'max_value': 100,
                'default_value': 0,
                'decimals': 3
            },
            {
                'attribute_name': 'k_upper_rt_spin',
                'label_text': 'Upper k RT:',
                'tooltip_text': 'Upper bound for k of an EMG chromatographic peak (affects peak skewness, must be higher than k_lower_rt).',
                'min_value': 0.01,
                'max_value': 100,
                'default_value': 10,
                'decimals': 3
            },
            {
                'attribute_name': 'k_alpha_rt_spin',
                'label_text': 'Alpha k RT:',
                'tooltip_text': 'Alpha for beta distribution for k_hat that is then scaled to k in (k_lower_rt, k_upper_rt).',
                'min_value': 0.01,
                'max_value': 100,
                'default_value': 1,
                'decimals': 3
            },
            {
                'attribute_name': 'k_beta_rt_spin',
                'label_text': 'Beta k RT:',
                'tooltip_text': 'Beta for beta distribution for k_hat that is then scaled to k in (k_lower_rt, k_upper_rt).',
                'min_value': 0.01,
                'max_value': 100,
                'default_value': 20,
                'decimals': 3
            },
            {
                'attribute_name': 'z_score_spin',
                'label_text': 'Z-Score:',
                'tooltip_text': 'Z-score threshold for filtering low-signal regions.',
                'min_value': 0,
                'max_value': 5,
                'default_value': 0.99,
                'decimals': 2
            },
            {
                'attribute_name': 'target_p_spin',
                'label_text': 'Target Percentile:',
                'tooltip_text': 'Target percentile for high-density regions in the data.',
                'min_value': 0,
                'max_value': 1,
                'default_value': 0.999,
                'decimals': 3
            },
            {
                'attribute_name': 'sampling_step_size_spin',
                'label_text': 'Sampling Step Size:',
                'tooltip_text': 'Step size for data sampling (affects resolution and speed).',
                'min_value': 0,
                'max_value': 1,
                'default_value': 0.001,
                'decimals': 4
            },
        ]
        for setting in settings:
            self.add_spinbox_with_info(settings_layout,
                                       attribute_name=setting['attribute_name'],
                                       label_text=setting['label_text'],
                                       tooltip_text=setting['tooltip_text'],
                                       min_value=setting['min_value'],
                                       max_value=setting['max_value'],
                                       default_value=setting['default_value'],
                                       decimals=setting.get('decimals', 2),
                                       step=setting.get('step', None))
        self.emg_plot = EMGPlot()
        plot_layout.addWidget(self.emg_plot)
        main_layout.addLayout(settings_layout)
        main_layout.addLayout(plot_layout)
        self.distribution_settings_group.content_layout.addLayout(main_layout)
        self.gradient_length_spin.valueChanged.connect(self.update_gradient_length)
        self.sigma_lower_rt_spin.valueChanged.connect(self.update_emg_sigma_lower)
        self.sigma_upper_rt_spin.valueChanged.connect(self.update_emg_sigma_upper)
        self.sigma_alpha_rt_spin.valueChanged.connect(self.update_emg_sigma_alpha)
        self.sigma_beta_rt_spin.valueChanged.connect(self.update_emg_sigma_beta)
        self.k_lower_rt_spin.valueChanged.connect(self.update_emg_k_lower)
        self.k_upper_rt_spin.valueChanged.connect(self.update_emg_k_upper)
        self.k_alpha_rt_spin.valueChanged.connect(self.update_emg_k_alpha)
        self.k_beta_rt_spin.valueChanged.connect(self.update_emg_k_beta)
        self.main_layout.addWidget(self.distribution_settings_group)

    def update_gradient_length(self, value):
        # update all emg parameters based on gradient length
        # get all defaults
        param_dict = calculate_rt_defaults(value)
        self.sigma_lower_rt_spin.setValue(param_dict['sigma_lower_rt'])
        self.sigma_upper_rt_spin.setValue(param_dict['sigma_upper_rt'])

    def update_emg_sigma_lower(self, value):
        self.emg_plot.set_sigma_lower(value)

    def update_emg_sigma_upper(self, value):
        self.emg_plot.set_sigma_upper(value)

    def update_emg_sigma_alpha(self, value):
        self.emg_plot.set_sigma_alpha(value)

    def update_emg_sigma_beta(self, value):
        self.emg_plot.set_sigma_beta(value)

    def update_emg_k_lower(self, value):
        self.emg_plot.set_k_lower(value)

    def update_emg_k_upper(self, value):
        self.emg_plot.set_k_upper(value)

    def update_emg_k_alpha(self, value):
        self.emg_plot.set_k_alpha(value)

    def update_emg_k_beta(self, value):
        self.emg_plot.set_k_beta(value)

    def init_noise_settings(self):
        info_text = "Configure simulation noise parameters, including m/z noise and real data noise options."
        self.noise_settings_group = CollapsibleBox("Noise Settings", info_text)
        layout = self.noise_settings_group.content_layout
        info_icon = QPixmap(str(self.script_dir / "info_icon.png")).scaled(19, 19, Qt.KeepAspectRatio)
        # Add Noise to Frame Abundance
        self.noise_frame_abundance_checkbox = QCheckBox("Add Noise to Frame Abundance")
        self.noise_frame_abundance_checkbox.setChecked(False)
        self.add_checkbox_with_info(layout, self.noise_frame_abundance_checkbox,
                                    "Add noise to frame abundance (default: False)")

        # Add Noise to Scan Abundance
        self.noise_scan_abundance_checkbox = QCheckBox("Add Noise to Scan Abundance")
        self.noise_scan_abundance_checkbox.setChecked(False)
        self.add_checkbox_with_info(layout, self.noise_scan_abundance_checkbox,
                                    "Add noise to scan abundance (default: False)")
        mz_noise_precursor_layout = QHBoxLayout()
        self.mz_noise_precursor_checkbox = QCheckBox("Add Precursor M/Z Noise")
        self.mz_noise_precursor_checkbox.setChecked(True)
        mz_noise_precursor_layout.addWidget(self.mz_noise_precursor_checkbox)
        mz_noise_precursor_info = QLabel()
        mz_noise_precursor_info.setPixmap(info_icon)
        mz_noise_precursor_info.setToolTip("Add noise to precursor m/z values.")
        mz_noise_precursor_layout.addWidget(mz_noise_precursor_info)
        layout.addLayout(mz_noise_precursor_layout)
        precursor_noise_ppm_layout = QHBoxLayout()
        self.precursor_noise_ppm_label = QLabel("Precursor Noise PPM:")
        self.precursor_noise_ppm_spin = QDoubleSpinBox()
        self.precursor_noise_ppm_spin.setRange(0, 100)
        self.precursor_noise_ppm_spin.setValue(5.0)
        precursor_noise_ppm_layout.addWidget(self.precursor_noise_ppm_label)
        precursor_noise_ppm_layout.addWidget(self.precursor_noise_ppm_spin)
        precursor_noise_ppm_info = QLabel()
        precursor_noise_ppm_info.setPixmap(info_icon)
        precursor_noise_ppm_info.setToolTip("Noise level (ppm) for precursor m/z values.")
        precursor_noise_ppm_layout.addWidget(precursor_noise_ppm_info)
        layout.addLayout(precursor_noise_ppm_layout)
        mz_noise_fragment_layout = QHBoxLayout()
        self.mz_noise_fragment_checkbox = QCheckBox("Add Fragment M/Z Noise")
        self.mz_noise_fragment_checkbox.setChecked(True)
        mz_noise_fragment_layout.addWidget(self.mz_noise_fragment_checkbox)
        mz_noise_fragment_info = QLabel()
        mz_noise_fragment_info.setPixmap(info_icon)
        mz_noise_fragment_info.setToolTip("Add noise to fragment m/z values.")
        mz_noise_fragment_layout.addWidget(mz_noise_fragment_info)
        layout.addLayout(mz_noise_fragment_layout)
        fragment_noise_ppm_layout = QHBoxLayout()
        self.fragment_noise_ppm_label = QLabel("Fragment Noise PPM:")
        self.fragment_noise_ppm_spin = QDoubleSpinBox()
        self.fragment_noise_ppm_spin.setRange(0, 100)
        self.fragment_noise_ppm_spin.setValue(5.0)
        fragment_noise_ppm_layout.addWidget(self.fragment_noise_ppm_label)
        fragment_noise_ppm_layout.addWidget(self.fragment_noise_ppm_spin)
        fragment_noise_ppm_info = QLabel()
        fragment_noise_ppm_info.setPixmap(info_icon)
        fragment_noise_ppm_info.setToolTip("Noise level (ppm) for fragment m/z values.")
        fragment_noise_ppm_layout.addWidget(fragment_noise_ppm_info)
        layout.addLayout(fragment_noise_ppm_layout)
        mz_noise_uniform_layout = QHBoxLayout()
        self.mz_noise_uniform_checkbox = QCheckBox("Use Uniform Distribution for M/Z Noise")
        self.mz_noise_uniform_checkbox.setChecked(False)
        mz_noise_uniform_layout.addWidget(self.mz_noise_uniform_checkbox)
        mz_noise_uniform_info = QLabel()
        mz_noise_uniform_info.setPixmap(info_icon)
        mz_noise_uniform_info.setToolTip("Use uniform instead of Gaussian distribution for m/z noise.")
        mz_noise_uniform_layout.addWidget(mz_noise_uniform_info)
        layout.addLayout(mz_noise_uniform_layout)
        real_data_noise_layout = QHBoxLayout()
        self.add_real_data_noise_checkbox = QCheckBox("Add Real Data Noise")
        self.add_real_data_noise_checkbox.setChecked(True)
        real_data_noise_layout.addWidget(self.add_real_data_noise_checkbox)
        real_data_noise_info = QLabel()
        real_data_noise_info.setPixmap(info_icon)
        real_data_noise_info.setToolTip("Add noise based on real data samples.")
        real_data_noise_layout.addWidget(real_data_noise_info)
        layout.addLayout(real_data_noise_layout)
        reference_noise_intensity_layout = QHBoxLayout()
        self.reference_noise_intensity_max_label = QLabel("Reference Noise Intensity Max:")
        self.reference_noise_intensity_max_spin = QDoubleSpinBox()
        self.reference_noise_intensity_max_spin.setRange(1, 1e5)
        self.reference_noise_intensity_max_spin.setValue(75)
        reference_noise_intensity_layout.addWidget(self.reference_noise_intensity_max_label)
        reference_noise_intensity_layout.addWidget(self.reference_noise_intensity_max_spin)
        reference_noise_intensity_info = QLabel()
        reference_noise_intensity_info.setPixmap(info_icon)
        reference_noise_intensity_info.setToolTip("Maximum noise intensity from reference data samples.")
        reference_noise_intensity_layout.addWidget(reference_noise_intensity_info)
        layout.addLayout(reference_noise_intensity_layout)
        downsample_factor_layout = QHBoxLayout()
        self.down_sample_factor_label = QLabel("Fragment Downsample Factor:")
        self.down_sample_factor_spin = QDoubleSpinBox()
        self.down_sample_factor_spin.setRange(0, 1)
        self.down_sample_factor_spin.setDecimals(2)
        self.down_sample_factor_spin.setValue(0.5)
        downsample_factor_layout.addWidget(self.down_sample_factor_label)
        downsample_factor_layout.addWidget(self.down_sample_factor_spin)
        downsample_factor_info = QLabel()
        downsample_factor_info.setPixmap(info_icon)
        downsample_factor_info.setToolTip("Downsample fragment ions based on intensity.")
        downsample_factor_layout.addWidget(downsample_factor_info)
        layout.addLayout(downsample_factor_layout)

    def init_charge_state_probabilities(self):
        info_text = "Set probabilities for peptide charge states."
        self.charge_state_probabilities_group = CollapsibleBox("Charge State Probabilities", info_text)
        layout = self.charge_state_probabilities_group.content_layout
        info_icon = QPixmap(str(self.script_dir / "info_icon.png")).scaled(19, 19, Qt.KeepAspectRatio)
        p_charge_layout = QHBoxLayout()
        self.p_charge_label = QLabel("Probability of Charge:")
        self.p_charge_spin = QDoubleSpinBox()
        self.p_charge_spin.setRange(0, 1)
        self.p_charge_spin.setDecimals(2)
        self.p_charge_spin.setValue(0.5)
        p_charge_layout.addWidget(self.p_charge_label)
        p_charge_layout.addWidget(self.p_charge_spin)
        p_charge_info = QLabel()
        p_charge_info.setPixmap(info_icon)
        p_charge_info.setToolTip("Probability for a peptide to adopt a given charge state.")
        p_charge_layout.addWidget(p_charge_info)
        layout.addLayout(p_charge_layout)
        min_charge_contrib_layout = QHBoxLayout()
        self.min_charge_contrib_label = QLabel("Minimum Charge Contribution:")
        self.min_charge_contrib_spin = QDoubleSpinBox()
        self.min_charge_contrib_spin.setRange(0, 1)
        self.min_charge_contrib_spin.setDecimals(2)
        self.min_charge_contrib_spin.setValue(0.25)
        min_charge_contrib_layout.addWidget(self.min_charge_contrib_label)
        min_charge_contrib_layout.addWidget(self.min_charge_contrib_spin)
        min_charge_contrib_info = QLabel()
        min_charge_contrib_info.setPixmap(info_icon)
        min_charge_contrib_info.setToolTip("Minimum relative contribution of peptides with this charge state.")
        min_charge_contrib_layout.addWidget(min_charge_contrib_info)
        layout.addLayout(min_charge_contrib_layout)

    def init_performance_settings(self):
        info_text = "Adjust performance settings such as threading and batch size."
        self.performance_settings_group = CollapsibleBox("Performance Settings", info_text)
        layout = self.performance_settings_group.content_layout
        info_icon = QPixmap(str(self.script_dir / "info_icon.png")).scaled(19, 19, Qt.KeepAspectRatio)
        num_threads_layout = QHBoxLayout()
        self.num_threads_label = QLabel("Number of Threads:")
        self.num_threads_spin = QSpinBox()
        self.num_threads_spin.setRange(-1, 64)
        self.num_threads_spin.setValue(-1)
        num_threads_layout.addWidget(self.num_threads_label)
        num_threads_layout.addWidget(self.num_threads_spin)
        num_threads_info = QLabel()
        num_threads_info.setPixmap(info_icon)
        num_threads_info.setToolTip("Number of threads to use (-1 uses all available cores).")
        num_threads_layout.addWidget(num_threads_info)
        layout.addLayout(num_threads_layout)
        batch_size_layout = QHBoxLayout()
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10000)
        self.batch_size_spin.setValue(256)
        batch_size_layout.addWidget(self.batch_size_label)
        batch_size_layout.addWidget(self.batch_size_spin)
        batch_size_info = QLabel()
        batch_size_info.setPixmap(info_icon)
        batch_size_info.setToolTip("Size of each processing batch.")
        batch_size_layout.addWidget(batch_size_info)
        layout.addLayout(batch_size_layout)
        self.run_button = QPushButton("RUN Simulation")
        self.run_button.setFixedHeight(50)
        self.run_button.setFont(QFont('Arial', 14))
        self.run_button.setStyleSheet("background-color: #6a994e; color: white; border: none;")
        self.run_button.clicked.connect(self.run_simulation)
        self.cancel_button = QPushButton("CANCEL Simulation")
        self.cancel_button.setFixedHeight(50)
        self.cancel_button.setFont(QFont('Arial', 14))
        self.cancel_button.setStyleSheet("background-color: #bc4749; color: white; border: none;")
        self.cancel_button.clicked.connect(self.cancel_simulation)
        self.cancel_button.setVisible(False)

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
        self.console.setFixedHeight(500)
        self.console.setFont(QFont('Courier', 10))

    def run_simulation(self):
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
        phospho_mode = self.phospho_mode_checkbox.isChecked()

        num_sample_peptides = self.num_sample_peptides_spin.value()
        missed_cleavages = self.missed_cleavages_spin.value()
        min_len = self.min_len_spin.value()
        max_len = self.max_len_spin.value()
        cleave_at = self.cleave_at_input.text()
        restrict = self.restrict_input.text()
        modifications = self.mods_input.text()
        sample_occurrences = self.sample_occurrences_checkbox.isChecked()

        isotope_k = self.isotope_k_spin.value()
        isotope_min_intensity = self.isotope_min_intensity_spin.value()
        isotope_centroid = self.isotope_centroid_checkbox.isChecked()

        gradient_length = self.gradient_length_spin.value()
        sigma_lower_rt = self.sigma_lower_rt_spin.value()
        sigma_upper_rt = self.sigma_upper_rt_spin.value()
        sigma_alpha_rt = self.sigma_alpha_rt_spin.value()
        sigma_beta_rt = self.sigma_beta_rt_spin.value()
        k_lower_rt = self.k_lower_rt_spin.value()
        k_upper_rt = self.k_upper_rt_spin.value()
        k_alpha_rt = self.k_alpha_rt_spin.value()
        k_beta_rt = self.k_beta_rt_spin.value()
        z_score = self.z_score_spin.value()
        target_p = self.target_p_spin.value()
        sampling_step_size = self.sampling_step_size_spin.value()

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
        if proteome_mix:
            fasta_path = self.multi_fasta_input.text()
            multi_fasta_dilution_path = self.multi_fasta_dilution_input.text()

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
            "--isotope_k", str(isotope_k),
            "--isotope_min_intensity", str(isotope_min_intensity),
            "--gradient_length", str(gradient_length),
            "--sigma_lower_rt", str(sigma_lower_rt),
            "--sigma_upper_rt", str(sigma_upper_rt),
            "--sigma_alpha_rt", str(sigma_alpha_rt),
            "--sigma_beta_rt", str(sigma_beta_rt),
            "--k_lower_rt", str(k_lower_rt),
            "--k_upper_rt", str(k_upper_rt),
            "--k_alpha_rt", str(k_alpha_rt),
            "--k_beta_rt", str(k_beta_rt),
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
            "--existing_path", use_existing_path,
        ]
        if multi_fasta_dilution_path and proteome_mix:
            args.extend(["--multi_fasta_dilution", multi_fasta_dilution_path])
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
        if phospho_mode:
            args.append("--phospho_mode")
        if modifications:
            args.extend(["--modifications", modifications])

        args.extend([
            "--precursors_every", str(self.precursors_every_spin.value()),
            "--precursor_intensity_threshold", str(self.precursor_intensity_threshold_spin.value()),
            "--max_precursors", str(self.max_precursors_spin.value()),
            "--exclusion_width", str(self.exclusion_width_spin.value()),
            "--selection_mode", self.selection_mode_combo.currentText(),
        ])

        args = [str(arg) for arg in args]
        if self.process and self.process.state() == QProcess.Running:
            self.process.kill()
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("background-color: #6d6875; color: white; border: none;")
        self.cancel_button.setVisible(True)
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
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
        if self.process and self.process.state() == QProcess.Running:
            reply = QMessageBox.question(self, "Cancel Simulation",
                                         "Are you sure you want to cancel the simulation?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.process.kill()
                self.console.append("Simulation canceled by user.")
                self.process_finished()
            else:
                self.console.append("Cancellation aborted by user.")

    def closeEvent(self, event):
        if self.process and self.process.state() == QProcess.Running:
            reply = QMessageBox.question(self, "Exit Application",
                                         "A simulation is running. Terminate it?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.process.kill()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def save_config(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Config File", "",
                                                   "TOML Files (*.toml);;All Files (*)", options=options)
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
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Config File", "",
                                                   "TOML Files (*.toml);;All Files (*)", options=options)
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
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Modifications File", "",
                                                   "TOML Files (*.toml);;All Files (*)", options=options)
        if file_path:
            QMessageBox.information(self, "Success", "Modifications loaded successfully.")
            self.mods_input.setText(file_path)
        else:
            QMessageBox.warning(self, "Error", "Failed to load modifications file.")

    def collect_settings(self):
        config = {}
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
            'phospho_mode': self.phospho_mode_checkbox.isChecked(),
            'multi_fasta': self.multi_fasta_input.text(),
            'multi_fasta_dilution': self.multi_fasta_dilution_input.text(),
        }
        config['peptide_digestion'] = {
            'sample_occurrences': self.sample_occurrences_checkbox.isChecked(),
            'num_sample_peptides': self.num_sample_peptides_spin.value(),
            'missed_cleavages': self.missed_cleavages_spin.value(),
            'min_len': self.min_len_spin.value(),
            'max_len': self.max_len_spin.value(),
            'cleave_at': self.cleave_at_input.text(),
            'restrict': self.restrict_input.text(),
            'modifications': self.mods_input.text(),
        }
        config['isotopic_pattern'] = {
            'isotope_k': self.isotope_k_spin.value(),
            'isotope_min_intensity': self.isotope_min_intensity_spin.value(),
            'isotope_centroid': self.isotope_centroid_checkbox.isChecked(),
        }
        config['distribution_settings'] = {
            'gradient_length': self.gradient_length_spin.value(),
            'sigma_lower_rt': self.sigma_lower_rt_spin.value(),
            'sigma_upper_rt': self.sigma_upper_rt_spin.value(),
            'sigma_alpha_rt': self.sigma_alpha_rt_spin.value(),
            'sigma_beta_rt': self.sigma_beta_rt_spin.value(),
            'k_lower_rt': self.k_lower_rt_spin.value(),
            'k_upper_rt': self.k_upper_rt_spin.value(),
            'k_alpha_rt': self.k_alpha_rt_spin.value(),
            'k_beta_rt': self.k_beta_rt_spin.value(),
            'z_score': self.z_score_spin.value(),
            'target_p': self.target_p_spin.value(),
            'sampling_step_size': self.sampling_step_size_spin.value(),
        }
        config['noise_settings'] = {
            'mz_noise_precursor': self.mz_noise_precursor_checkbox.isChecked(),
            'precursor_noise_ppm': self.precursor_noise_ppm_spin.value(),
            'mz_noise_fragment': self.mz_noise_fragment_checkbox.isChecked(),
            'fragment_noise_ppm': self.fragment_noise_ppm_spin.value(),
            'mz_noise_uniform': self.mz_noise_uniform_checkbox.isChecked(),
            'add_real_data_noise': self.add_real_data_noise_checkbox.isChecked(),
            'reference_noise_intensity_max': self.reference_noise_intensity_max_spin.value(),
            'down_sample_factor': self.down_sample_factor_spin.value(),
            'noise_frame_abundance': self.noise_frame_abundance_checkbox.isChecked(),
            'noise_scan_abundance': self.noise_scan_abundance_checkbox.isChecked(),
        }
        config['charge_state_probabilities'] = {
            'p_charge': self.p_charge_spin.value(),
            'min_charge_contrib': self.min_charge_contrib_spin.value(),
        }
        config['dda_settings'] = {
            'precursors_every': self.precursors_every_spin.value(),
            'precursor_intensity_threshold': self.precursor_intensity_threshold_spin.value(),
            'max_precursors': self.max_precursors_spin.value(),
            'exclusion_width': self.exclusion_width_spin.value(),
            'selection_mode': self.selection_mode_combo.currentText()
        }
        config['performance_settings'] = {
            'num_threads': self.num_threads_spin.value(),
            'batch_size': self.batch_size_spin.value(),
        }
        return config

    def apply_settings(self, config):

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
        self.apply_fragmentation_checkbox.setChecked(main_settings.get('apply_fragmentation', False))
        self.phospho_mode_checkbox.setChecked(main_settings.get('phospho_mode', False))
        self.from_existing_checkbox.setChecked(main_settings.get('from_existing', False))
        self.existing_path_input.setText(main_settings.get('existing_path', ''))
        self.multi_fasta_input.setText(main_settings.get('multi_fasta', ''))
        self.multi_fasta_dilution_input.setText(main_settings.get('multi_fasta_dilution', ''))

        peptide_digestion = config.get('peptide_digestion', {})
        self.num_sample_peptides_spin.setValue(peptide_digestion.get('num_sample_peptides', 25000))
        self.missed_cleavages_spin.setValue(peptide_digestion.get('missed_cleavages', 2))
        self.min_len_spin.setValue(peptide_digestion.get('min_len', 7))
        self.max_len_spin.setValue(peptide_digestion.get('max_len', 30))
        self.cleave_at_input.setText(peptide_digestion.get('cleave_at', 'KR'))
        self.restrict_input.setText(peptide_digestion.get('restrict', 'P'))
        self.mods_input.setText(peptide_digestion.get('modifications', ''))

        self.sample_occurrences_checkbox.setChecked(peptide_digestion.get('sample_occurrences', True))

        isotopic_pattern = config.get('isotopic_pattern', {})
        self.isotope_k_spin.setValue(isotopic_pattern.get('isotope_k', 8))
        self.isotope_min_intensity_spin.setValue(isotopic_pattern.get('isotope_min_intensity', 1))
        self.isotope_centroid_checkbox.setChecked(isotopic_pattern.get('isotope_centroid', True))

        distribution_settings = config.get('distribution_settings', {})
        # TODO redefinition of defaults is bad practice
        self.gradient_length_spin.setValue(distribution_settings.get('gradient_length', 3600))
        self.sigma_lower_rt_spin.setValue(distribution_settings.get('sigma_lower_rt', 1.969))
        self.sigma_upper_rt_spin.setValue(distribution_settings.get('sigma_upper_rt', 3.281))
        self.sigma_alpha_rt_spin.setValue(distribution_settings.get('sigma_alpha_rt', 4.0))
        self.sigma_beta_rt_spin.setValue(distribution_settings.get('sigma_beta_rt', 4.0))
        self.k_lower_rt_spin.setValue(distribution_settings.get('k_lower_rt', 0.0))
        self.k_upper_rt_spin.setValue(distribution_settings.get('k_upper_rt', 10.0))
        self.k_alpha_rt_spin.setValue(distribution_settings.get('k_alpha_rt', 1.0))
        self.k_beta_rt_spin.setValue(distribution_settings.get('k_beta_rt', 20.0))
        self.z_score_spin.setValue(distribution_settings.get('z_score', 0.99))
        self.target_p_spin.setValue(distribution_settings.get('target_p', 0.999))
        self.sampling_step_size_spin.setValue(distribution_settings.get('sampling_step_size', 0.001))

        noise_settings = config.get('noise_settings', {})
        self.mz_noise_precursor_checkbox.setChecked(noise_settings.get('mz_noise_precursor', False))
        self.precursor_noise_ppm_spin.setValue(noise_settings.get('precursor_noise_ppm', 5.0))
        self.mz_noise_fragment_checkbox.setChecked(noise_settings.get('mz_noise_fragment', False))
        self.fragment_noise_ppm_spin.setValue(noise_settings.get('fragment_noise_ppm', 5.0))
        self.mz_noise_uniform_checkbox.setChecked(noise_settings.get('mz_noise_uniform', False))
        self.add_real_data_noise_checkbox.setChecked(noise_settings.get('add_real_data_noise', False))
        self.reference_noise_intensity_max_spin.setValue(noise_settings.get('reference_noise_intensity_max', 30))
        self.down_sample_factor_spin.setValue(noise_settings.get('down_sample_factor', 0.5))
        self.noise_frame_abundance_checkbox.setChecked(noise_settings.get('noise_frame_abundance', False))
        self.noise_scan_abundance_checkbox.setChecked(noise_settings.get('noise_scan_abundance', False))

        charge_state_probabilities = config.get('charge_state_probabilities', {})
        self.p_charge_spin.setValue(charge_state_probabilities.get('p_charge', 0.5))
        self.min_charge_contrib_spin.setValue(charge_state_probabilities.get('min_charge_contrib', 0.25))

        performance_settings = config.get('performance_settings', {})
        self.num_threads_spin.setValue(performance_settings.get('num_threads', -1))
        self.batch_size_spin.setValue(performance_settings.get('batch_size', 256))

        dda_settings = config.get('dda_settings', {})
        self.precursors_every_spin.setValue(dda_settings.get('precursors_every', 7))
        self.precursor_intensity_threshold_spin.setValue(dda_settings.get('precursor_intensity_threshold', 500))
        self.max_precursors_spin.setValue(dda_settings.get('max_precursors', 7))
        self.exclusion_width_spin.setValue(dda_settings.get('exclusion_width', 25))
        index = self.selection_mode_combo.findText(dda_settings.get('selection_mode', "topN"))
        if index >= 0:
            self.selection_mode_combo.setCurrentIndex(index)


def main():
    app = QApplication(sys.argv)
    script_dir = Path(__file__).resolve().parent.parent
    path = script_dir / "resources" / "icons" / "logo_2.png"
    print(path)
    app.setWindowIcon(QIcon(str(path)))
    app.setApplicationName("TimSim")
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()