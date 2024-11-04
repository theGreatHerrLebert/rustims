import sys
import time
from pathlib import Path
import qdarkstyle

import toml
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, QSpinBox, QToolButton,
    QDoubleSpinBox, QComboBox, QGroupBox, QScrollArea, QAction, QMessageBox,
    QTextEdit, QSlider
)
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QFont, QIcon, QPixmap

# CollapsibleBox class definition
class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.content_area = QWidget()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)

        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Create layout for content area
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(25, 0, 0, 0)  # Indent content
        self.content_area.setLayout(self.content_layout)

        # Main layout
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)
        self.setLayout(lay)

    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        if checked:
            # Expand
            self.content_area.setMaximumHeight(16777215)
            self.content_area.setMinimumHeight(0)
        else:
            # Collapse
            self.content_area.setMaximumHeight(0)
            self.content_area.setMinimumHeight(0)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

# MainWindow class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Path to the script directory
        self.script_dir = Path(__file__).parent.parent / "resources/icons/"

        print(self.script_dir)

        self.setWindowTitle("🦀💻 TimSim 🔬🐍 - Proteomics Experiment Simulation on timsTOF")
        self.setGeometry(100, 100, 800, 600)

        # Set the app logo

        self.setWindowIcon(QIcon(str(self.script_dir / "logo_2.png")))

        # Initialize the menu bar
        self.init_menu_bar()

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # Initialize sections
        self.init_main_settings()
        self.init_peptide_digestion_settings()
        self.init_peptide_intensity_settings()
        self.init_isotopic_pattern_settings()
        self.init_distribution_settings()
        self.init_noise_settings()
        self.init_charge_state_probabilities()
        self.init_performance_settings()
        self.init_console()

        # Add all sections to the main layout
        self.main_layout.addWidget(self.main_settings_group)
        self.main_layout.addWidget(self.peptide_digestion_group)
        self.main_layout.addWidget(self.peptide_intensity_group)
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
        scroll.setWidget(central_widget)
        self.setCentralWidget(scroll)
        
        # Initialize the process variable
        self.process = None

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

    def init_main_settings(self):
        self.main_settings_group = QGroupBox("Main Settings")
        layout = QVBoxLayout()

        # Experiment Paths
        path_layout = QHBoxLayout()
        self.path_label = QLabel("Save Path:")
        self.path_input = QLineEdit()
        self.path_browse = QPushButton("Browse")
        self.path_browse.clicked.connect(self.browse_save_path)
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.path_browse)
        layout.addLayout(path_layout)

        reference_layout = QHBoxLayout()
        self.reference_label = QLabel("Reference Dataset Path:")
        self.reference_input = QLineEdit()
        self.reference_browse = QPushButton("Browse")
        self.reference_browse.clicked.connect(self.browse_reference_path)
        reference_layout.addWidget(self.reference_label)
        reference_layout.addWidget(self.reference_input)
        reference_layout.addWidget(self.reference_browse)
        layout.addLayout(reference_layout)

        fasta_layout = QHBoxLayout()
        self.fasta_label = QLabel("FASTA File Path:")
        self.fasta_input = QLineEdit()
        self.fasta_browse = QPushButton("Browse")
        self.fasta_browse.clicked.connect(self.browse_fasta_path)
        fasta_layout.addWidget(self.fasta_label)
        fasta_layout.addWidget(self.fasta_input)
        fasta_layout.addWidget(self.fasta_browse)
        layout.addLayout(fasta_layout)

        # Experiment Details
        self.name_label = QLabel("Experiment Name:")
        self.name_input = QLineEdit(f"TIMSIM-{int(time.time())}")
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)

        self.acquisition_label = QLabel("Acquisition Type:")
        self.acquisition_combo = QComboBox()
        self.acquisition_combo.addItems(["DIA", "SYNCHRO", "SLICE", "MIDIA"])
        layout.addWidget(self.acquisition_label)
        layout.addWidget(self.acquisition_combo)

        # Checkboxes for options
        self.use_reference_layout_checkbox = QCheckBox("Use Reference Layout")
        self.use_reference_layout_checkbox.setChecked(True)
        layout.addWidget(self.use_reference_layout_checkbox)

        self.reference_in_memory_checkbox = QCheckBox("Load Reference into Memory")
        self.reference_in_memory_checkbox.setChecked(False)
        layout.addWidget(self.reference_in_memory_checkbox)

        self.sample_peptides_checkbox = QCheckBox("Sample Peptides")
        self.sample_peptides_checkbox.setChecked(True)
        layout.addWidget(self.sample_peptides_checkbox)

        self.add_decoys_checkbox = QCheckBox("Generate Decoys")
        self.add_decoys_checkbox.setChecked(False)
        layout.addWidget(self.add_decoys_checkbox)

        self.proteome_mix_checkbox = QCheckBox("Proteome Mixture")
        self.proteome_mix_checkbox.setChecked(False)
        layout.addWidget(self.proteome_mix_checkbox)

        self.silent_checkbox = QCheckBox("Silent Mode")
        self.silent_checkbox.setChecked(False)
        layout.addWidget(self.silent_checkbox)

        self.main_settings_group.setLayout(layout)

    # Update the initialization methods to use CollapsibleBox
    def init_peptide_digestion_settings(self):
        self.peptide_digestion_group = CollapsibleBox("Peptide Digestion Settings")
        layout = self.peptide_digestion_group.content_layout

        # Number of Sampled Peptides
        self.num_sample_peptides_label = QLabel("Number of Sampled Peptides:")
        self.num_sample_peptides_spin = QSpinBox()
        self.num_sample_peptides_spin.setRange(1, 1000000)
        self.num_sample_peptides_spin.setValue(25000)
        layout.addWidget(self.num_sample_peptides_label)
        layout.addWidget(self.num_sample_peptides_spin)

        # Missed Cleavages
        self.missed_cleavages_label = QLabel("Missed Cleavages:")
        self.missed_cleavages_spin = QSpinBox()
        self.missed_cleavages_spin.setRange(0, 10)
        self.missed_cleavages_spin.setValue(2)
        layout.addWidget(self.missed_cleavages_label)
        layout.addWidget(self.missed_cleavages_spin)

        # Minimum Peptide Length
        self.min_len_label = QLabel("Minimum Peptide Length:")
        self.min_len_spin = QSpinBox()
        self.min_len_spin.setRange(1, 100)
        self.min_len_spin.setValue(7)
        layout.addWidget(self.min_len_label)
        layout.addWidget(self.min_len_spin)

        # Maximum Peptide Length
        self.max_len_label = QLabel("Maximum Peptide Length:")
        self.max_len_spin = QSpinBox()
        self.max_len_spin.setRange(1, 100)
        self.max_len_spin.setValue(30)
        layout.addWidget(self.max_len_label)
        layout.addWidget(self.max_len_spin)

        # Cleave At
        self.cleave_at_label = QLabel("Cleave At:")
        self.cleave_at_input = QLineEdit("KR")
        layout.addWidget(self.cleave_at_label)
        layout.addWidget(self.cleave_at_input)

        # Restrict
        self.restrict_label = QLabel("Restrict:")
        self.restrict_input = QLineEdit("P")
        layout.addWidget(self.restrict_label)
        layout.addWidget(self.restrict_input)

    def init_peptide_intensity_settings(self):
        self.peptide_intensity_group = CollapsibleBox("Peptide Intensity Settings")
        layout = self.peptide_intensity_group.content_layout

        # Function to format slider value as 10^power
        def update_label_from_slider(slider, label):
            power = slider.value()
            label.setText(f"10^{power} ({10 ** power:.1e})")

        # Create slider for Mean Intensity
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

        # Create slider for Minimum Intensity
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

        # Create slider for Maximum Intensity
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

        # Sample Occurrences Randomly
        self.sample_occurrences_checkbox = QCheckBox("Sample Occurrences Randomly")
        self.sample_occurrences_checkbox.setChecked(True)
        layout.addWidget(self.sample_occurrences_checkbox)

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

    def init_isotopic_pattern_settings(self):
        self.isotopic_pattern_group = CollapsibleBox("Isotopic Pattern Settings")
        layout = self.isotopic_pattern_group.content_layout

        # Number of Isotopes
        self.isotope_k_label = QLabel("Number of Isotopes:")
        self.isotope_k_spin = QSpinBox()
        self.isotope_k_spin.setRange(1, 20)
        self.isotope_k_spin.setValue(8)
        layout.addWidget(self.isotope_k_label)
        layout.addWidget(self.isotope_k_spin)

        # Minimum Isotope Intensity
        self.isotope_min_intensity_label = QLabel("Minimum Isotope Intensity:")
        self.isotope_min_intensity_spin = QSpinBox()
        self.isotope_min_intensity_spin.setRange(0, 1000)
        self.isotope_min_intensity_spin.setValue(1)
        layout.addWidget(self.isotope_min_intensity_label)
        layout.addWidget(self.isotope_min_intensity_spin)

        # Centroid Isotopes
        self.isotope_centroid_checkbox = QCheckBox("Centroid Isotopes")
        self.isotope_centroid_checkbox.setChecked(True)
        layout.addWidget(self.isotope_centroid_checkbox)

    def init_distribution_settings(self):
        self.distribution_settings_group = CollapsibleBox("Signal distribution Settings")
        layout = self.distribution_settings_group.content_layout

        # Gradient Length
        self.gradient_length_label = QLabel("Gradient Length (seconds):")
        self.gradient_length_spin = QDoubleSpinBox()
        self.gradient_length_spin.setRange(1, 1e5)
        self.gradient_length_spin.setValue(3600)
        layout.addWidget(self.gradient_length_label)
        layout.addWidget(self.gradient_length_spin)

        # Mean Std RT
        self.mean_std_rt_label = QLabel("Mean Std RT:")
        self.mean_std_rt_spin = QDoubleSpinBox()
        self.mean_std_rt_spin.setRange(0.1, 10)
        self.mean_std_rt_spin.setValue(1.5)
        layout.addWidget(self.mean_std_rt_label)
        layout.addWidget(self.mean_std_rt_spin)

        # Variance Std RT
        self.variance_std_rt_label = QLabel("Variance Std RT:")
        self.variance_std_rt_spin = QDoubleSpinBox()
        self.variance_std_rt_spin.setRange(0, 5)
        self.variance_std_rt_spin.setValue(0.3)
        layout.addWidget(self.variance_std_rt_label)
        layout.addWidget(self.variance_std_rt_spin)

        # Mean Skewness
        self.mean_skewness_label = QLabel("Mean Skewness:")
        self.mean_skewness_spin = QDoubleSpinBox()
        self.mean_skewness_spin.setRange(-5, 5)
        self.mean_skewness_spin.setValue(0.3)
        layout.addWidget(self.mean_skewness_label)
        layout.addWidget(self.mean_skewness_spin)

        # Variance Skewness
        self.variance_skewness_label = QLabel("Variance Skewness:")
        self.variance_skewness_spin = QDoubleSpinBox()
        self.variance_skewness_spin.setRange(0, 5)
        self.variance_skewness_spin.setValue(0.1)
        layout.addWidget(self.variance_skewness_label)
        layout.addWidget(self.variance_skewness_spin)

        # Standard Deviation IM
        self.std_im_label = QLabel("Standard Deviation IM:")
        self.std_im_spin = QDoubleSpinBox()
        self.std_im_spin.setRange(0, 1)
        self.std_im_spin.setDecimals(4)
        self.std_im_spin.setValue(0.01)
        layout.addWidget(self.std_im_label)
        layout.addWidget(self.std_im_spin)

        # Z-Score
        self.z_score_label = QLabel("Z-Score:")
        self.z_score_spin = QDoubleSpinBox()
        self.z_score_spin.setRange(0, 5)
        self.z_score_spin.setValue(0.99)
        layout.addWidget(self.z_score_label)
        layout.addWidget(self.z_score_spin)

        # Target Percentile
        self.target_p_label = QLabel("Target Percentile:")
        self.target_p_spin = QDoubleSpinBox()
        self.target_p_spin.setRange(0, 1)
        self.target_p_spin.setDecimals(3)
        self.target_p_spin.setValue(0.999)
        layout.addWidget(self.target_p_label)
        layout.addWidget(self.target_p_spin)

        # Sampling Step Size
        self.sampling_step_size_label = QLabel("Sampling Step Size:")
        self.sampling_step_size_spin = QDoubleSpinBox()
        self.sampling_step_size_spin.setRange(0, 1)
        self.sampling_step_size_spin.setDecimals(4)
        self.sampling_step_size_spin.setValue(0.001)
        layout.addWidget(self.sampling_step_size_label)
        layout.addWidget(self.sampling_step_size_spin)

    def init_noise_settings(self):
        self.noise_settings_group = CollapsibleBox("Noise Settings")
        layout = self.noise_settings_group.content_layout

        # Add Noise to Signals
        self.add_noise_to_signals_checkbox = QCheckBox("Add Noise to Signals")
        self.add_noise_to_signals_checkbox.setChecked(True)
        layout.addWidget(self.add_noise_to_signals_checkbox)

        # Add Precursor M/Z Noise
        self.mz_noise_precursor_checkbox = QCheckBox("Add Precursor M/Z Noise")
        self.mz_noise_precursor_checkbox.setChecked(True)
        layout.addWidget(self.mz_noise_precursor_checkbox)

        # Precursor Noise PPM
        self.precursor_noise_ppm_label = QLabel("Precursor Noise PPM:")
        self.precursor_noise_ppm_spin = QDoubleSpinBox()
        self.precursor_noise_ppm_spin.setRange(0, 100)
        self.precursor_noise_ppm_spin.setValue(5.0)
        layout.addWidget(self.precursor_noise_ppm_label)
        layout.addWidget(self.precursor_noise_ppm_spin)

        # Add Fragment M/Z Noise
        self.mz_noise_fragment_checkbox = QCheckBox("Add Fragment M/Z Noise")
        self.mz_noise_fragment_checkbox.setChecked(True)
        layout.addWidget(self.mz_noise_fragment_checkbox)

        # Fragment Noise PPM
        self.fragment_noise_ppm_label = QLabel("Fragment Noise PPM:")
        self.fragment_noise_ppm_spin = QDoubleSpinBox()
        self.fragment_noise_ppm_spin.setRange(0, 100)
        self.fragment_noise_ppm_spin.setValue(10.0)
        layout.addWidget(self.fragment_noise_ppm_label)
        layout.addWidget(self.fragment_noise_ppm_spin)

        # Use Uniform Distribution
        self.mz_noise_uniform_checkbox = QCheckBox("Use Uniform Distribution for M/Z Noise")
        self.mz_noise_uniform_checkbox.setChecked(False)
        layout.addWidget(self.mz_noise_uniform_checkbox)

        # Add Real Data Noise
        self.add_real_data_noise_checkbox = QCheckBox("Add Real Data Noise")
        self.add_real_data_noise_checkbox.setChecked(True)
        layout.addWidget(self.add_real_data_noise_checkbox)

        # Reference Noise Intensity Max
        self.reference_noise_intensity_max_label = QLabel("Reference Noise Intensity Max:")
        self.reference_noise_intensity_max_spin = QDoubleSpinBox()
        self.reference_noise_intensity_max_spin.setRange(1, 1e5)
        self.reference_noise_intensity_max_spin.setValue(75)
        layout.addWidget(self.reference_noise_intensity_max_label)
        layout.addWidget(self.reference_noise_intensity_max_spin)

        # Downsample Factor
        self.down_sample_factor_label = QLabel("Downsample Factor:")
        self.down_sample_factor_spin = QDoubleSpinBox()
        self.down_sample_factor_spin.setRange(0, 1)
        self.down_sample_factor_spin.setDecimals(2)
        self.down_sample_factor_spin.setValue(0.5)
        layout.addWidget(self.down_sample_factor_label)
        layout.addWidget(self.down_sample_factor_spin)

    def init_charge_state_probabilities(self):
        self.charge_state_probabilities_group = CollapsibleBox("Charge State Probabilities")
        layout = self.charge_state_probabilities_group.content_layout

        # Probability of Charge
        self.p_charge_label = QLabel("Probability of Charge:")
        self.p_charge_spin = QDoubleSpinBox()
        self.p_charge_spin.setRange(0, 1)
        self.p_charge_spin.setDecimals(2)
        self.p_charge_spin.setValue(0.5)
        layout.addWidget(self.p_charge_label)
        layout.addWidget(self.p_charge_spin)

        # Minimum Charge Contribution
        self.min_charge_contrib_label = QLabel("Minimum Charge Contribution:")
        self.min_charge_contrib_spin = QDoubleSpinBox()
        self.min_charge_contrib_spin.setRange(0, 1)
        self.min_charge_contrib_spin.setDecimals(2)
        self.min_charge_contrib_spin.setValue(0.25)
        layout.addWidget(self.min_charge_contrib_label)
        layout.addWidget(self.min_charge_contrib_spin)

    def init_performance_settings(self):
        self.performance_settings_group = CollapsibleBox("Performance Settings")
        layout = self.performance_settings_group.content_layout

        # Number of Threads
        self.num_threads_label = QLabel("Number of Threads:")
        self.num_threads_spin = QSpinBox()
        self.num_threads_spin.setRange(-1, 64)
        self.num_threads_spin.setValue(-1)
        layout.addWidget(self.num_threads_label)
        layout.addWidget(self.num_threads_spin)

        # Batch Size
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10000)
        self.batch_size_spin.setValue(256)
        layout.addWidget(self.batch_size_label)
        layout.addWidget(self.batch_size_spin)

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

    def init_console(self):
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(500)  # Adjust the height as needed
        self.console.setFont(QFont('Courier', 10))  # Use a monospace font for console output

    def run_simulation(self):
        # Collect all parameters and run the simulation
        # This function will map GUI inputs to command-line arguments

        # Example of collecting parameters from Main Settings
        save_path = self.path_input.text()
        reference_path = self.reference_input.text()
        fasta_path = self.fasta_input.text()
        experiment_name = self.name_input.text()
        acquisition_type = self.acquisition_combo.currentText()
        use_reference_layout = self.use_reference_layout_checkbox.isChecked()
        reference_in_memory = self.reference_in_memory_checkbox.isChecked()
        sample_peptides = self.sample_peptides_checkbox.isChecked()
        add_decoys = self.add_decoys_checkbox.isChecked()
        proteome_mix = self.proteome_mix_checkbox.isChecked()
        silent_mode = self.silent_checkbox.isChecked()

        # Collect other parameters
        num_sample_peptides = self.num_sample_peptides_spin.value()
        missed_cleavages = self.missed_cleavages_spin.value()
        min_len = self.min_len_spin.value()
        max_len = self.max_len_spin.value()
        cleave_at = self.cleave_at_input.text()
        restrict = self.restrict_input.text()

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

        # Build the argument list
        args = [
            "timsim",
            save_path,
            reference_path,
            fasta_path,
            "--name", experiment_name,
            "--acquisition_type", acquisition_type,
            "--num_sample_peptides", str(num_sample_peptides),
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
            "--mean_scewness", str(mean_skewness),
            "--variance_scewness", str(variance_skewness),
            "--std_im", str(std_im),
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
        ]

        # Add boolean flags
        if not use_reference_layout:
            args.append("--no_reference_layout")
        if reference_in_memory:
            args.append("--reference_in_memory")
        if not sample_peptides:
            args.append("--no_peptide_sampling")
        if add_decoys:
            args.append("--add_decoys")
        if proteome_mix:
            args.append("--proteome_mix")
        if silent_mode:
            args.append("--silent")
        if not sample_occurrences:
            args.append("--no_sample_occurrences")
        if isotope_centroid:
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
            'add_decoys': self.add_decoys_checkbox.isChecked(),
            'proteome_mix': self.proteome_mix_checkbox.isChecked(),
            'silent_mode': self.silent_checkbox.isChecked(),
        }

        # Peptide Digestion Settings
        config['peptide_digestion'] = {
            'num_sample_peptides': self.num_sample_peptides_spin.value(),
            'missed_cleavages': self.missed_cleavages_spin.value(),
            'min_len': self.min_len_spin.value(),
            'max_len': self.max_len_spin.value(),
            'cleave_at': self.cleave_at_input.text(),
            'restrict': self.restrict_input.text(),
        }

        # Peptide Intensity Settings
        config['peptide_intensity'] = {
            'intensity_mean': 10 ** self.intensity_mean_slider.value(),
            'intensity_min': 10 ** self.intensity_min_slider.value(),
            'intensity_max': 10 ** self.intensity_max_slider.value(),
            'sample_occurrences': self.sample_occurrences_checkbox.isChecked(),
            'intensity_value': 10 ** self.intensity_value_slider.value(),
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
        self.add_decoys_checkbox.setChecked(main_settings.get('add_decoys', False))
        self.proteome_mix_checkbox.setChecked(main_settings.get('proteome_mix', False))
        self.silent_checkbox.setChecked(main_settings.get('silent_mode', False))

        # Peptide Digestion Settings
        peptide_digestion = config.get('peptide_digestion', {})
        self.num_sample_peptides_spin.setValue(peptide_digestion.get('num_sample_peptides', 25000))
        self.missed_cleavages_spin.setValue(peptide_digestion.get('missed_cleavages', 2))
        self.min_len_spin.setValue(peptide_digestion.get('min_len', 7))
        self.max_len_spin.setValue(peptide_digestion.get('max_len', 30))
        self.cleave_at_input.setText(peptide_digestion.get('cleave_at', 'KR'))
        self.restrict_input.setText(peptide_digestion.get('restrict', 'P'))

        # Peptide Intensity Settings
        peptide_intensity = config.get('peptide_intensity', {})
        self.intensity_mean_spin.setValue(peptide_intensity.get('intensity_mean', 1e7))
        self.intensity_min_spin.setValue(peptide_intensity.get('intensity_min', 1e5))
        self.intensity_max_spin.setValue(peptide_intensity.get('intensity_max', 1e9))
        self.sample_occurrences_checkbox.setChecked(peptide_intensity.get('sample_occurrences', True))
        self.intensity_value_spin.setValue(peptide_intensity.get('intensity_value', 1e6))

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