"""PySide workbench over the shared CLI, example renderer, and registry."""
from __future__ import annotations

from pathlib import Path


def launch_gui() -> int:
    try:
        from PySide6.QtCore import QObject, QProcess, QRunnable, QThreadPool, QTimer, Qt, QUrl, Signal
        from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
        from PySide6.QtWidgets import (
            QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
            QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
            QSizePolicy, QSpinBox, QStyle, QTabWidget, QTableWidget, QTableWidgetItem,
            QTextEdit, QVBoxLayout, QWidget,
        )
    except ImportError as exc:
        raise RuntimeError('GUI support requires: pip install -e ".[gui]"') from exc

    class WorkerSignals(QObject):
        result = Signal(object)
        error = Signal(str)
        finished = Signal()

    class Worker(QRunnable):
        """Run a non-Qt callable without blocking the GUI event loop."""
        def __init__(self, function: object) -> None:
            super().__init__(); self.function = function; self.signals = WorkerSignals()

        def run(self) -> None:
            try:
                result = self.function()
            except Exception as exc:
                try:
                    self.signals.error.emit(str(exc))
                except RuntimeError:
                    pass  # The application was closed while work was finishing.
            else:
                try:
                    self.signals.result.emit(result)
                except RuntimeError:
                    pass
            finally:
                try:
                    self.signals.finished.emit()
                except RuntimeError:
                    pass

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("SSVEP spike encoding workbench")
            icon = QIcon(str(Path(__file__).with_name("assets") / "app_icon.svg"))
            self.setWindowIcon(icon)
            self.resize(1180, 780)
            self.process = QProcess(self)
            self.current_run_log_path = None
            self.example_process = QProcess(self); self.example_pending = False
            self.dashboard_process = QProcess(self); self.dashboard_open_after = False
            self.thread_pool = QThreadPool.globalInstance(); self.background_workers = []
            self.run_output_buffer = ""; self.refresh_active = False; self.cleanup_active = False
            self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)
            self._build_explorer_tab()
            self._build_run_tab()
            self._build_dashboard_tab()
            self.process.errorOccurred.connect(self.experiment_process_error)
            self.example_process.errorOccurred.connect(self.example_process_error)
            self.dashboard_process.errorOccurred.connect(self.dashboard_process_error)

        def _folder_row(self, field: QLineEdit) -> QWidget:
            widget = QWidget(); row = QHBoxLayout(widget); row.setContentsMargins(0, 0, 0, 0)
            button = QPushButton("Browse…")
            def browse() -> None:
                selected = QFileDialog.getExistingDirectory(self, "Choose folder", field.text())
                if selected:
                    field.setText(selected)
            button.clicked.connect(browse); row.addWidget(field); row.addWidget(button)
            return widget

        def _file_row(self, field: QLineEdit, pattern: str) -> QWidget:
            widget = QWidget(); row = QHBoxLayout(widget); row.setContentsMargins(0, 0, 0, 0)
            button = QPushButton("Browse…")
            def browse() -> None:
                selected, _ = QFileDialog.getOpenFileName(self, "Choose file", field.text(), pattern)
                if selected:
                    field.setText(selected)
            button.clicked.connect(browse); row.addWidget(field); row.addWidget(button)
            return widget

        def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
            control = QSpinBox(); control.setRange(minimum, maximum); control.setValue(value); return control

        def _double(self, minimum: float, maximum: float, value: float,
                    decimals: int = 3, step: float = 0.1) -> QDoubleSpinBox:
            control = QDoubleSpinBox(); control.setRange(minimum, maximum)
            control.setDecimals(decimals); control.setSingleStep(step); control.setValue(value); return control

        @staticmethod
        def _progress_idle(progress: QProgressBar, text: str = "Ready") -> None:
            progress.setRange(0, 1); progress.setValue(1); progress.setFormat(text)

        @staticmethod
        def _progress_busy(progress: QProgressBar, text: str) -> None:
            progress.setRange(0, 0); progress.setFormat(text)

        def _run_worker(self, function: object, on_result: object,
                        on_error: object, on_finished: object) -> None:
            worker = Worker(function); self.background_workers.append(worker)
            worker.signals.result.connect(on_result); worker.signals.error.connect(on_error)
            def finished() -> None:
                if worker in self.background_workers:
                    self.background_workers.remove(worker)
                on_finished()
            worker.signals.finished.connect(finished)
            self.thread_pool.start(worker)

        def _build_explorer_tab(self) -> None:
            tab = QWidget(); layout = QVBoxLayout(tab); form = QFormLayout()
            self.example_data = QLineEdit(str(Path.cwd().parent))
            self.example_output = QLineEdit(str(Path("outputs/examples/neuron_behavior")))
            self.example_subject = self._spin(1, 30, 1)
            self.example_frequency = QComboBox(); self.example_frequency.addItems(tuple(str(x) for x in range(1, 61)))
            self.example_frequency.setCurrentText("8")
            self.example_block = self._spin(1, 12, 1)
            self.example_electrode = QComboBox(); self.example_electrode.addItems(("O1", "Oz", "O2", "O1-Oz", "O2-Oz")); self.example_electrode.setCurrentText("Oz")
            self.example_encoder = QComboBox(); self.example_encoder.addItems(("All encoders", "Resonate-and-fire", "Delta", "LIF"))
            self.example_auto = QCheckBox("Update automatically when the selection changes")
            self.example_auto.setChecked(True)
            form.addRow("Dataset folder", self._folder_row(self.example_data))
            form.addRow("Example output folder", self._folder_row(self.example_output))
            selectors = QWidget(); selector_row = QHBoxLayout(selectors); selector_row.setContentsMargins(0, 0, 0, 0)
            for label, control in (("Subject", self.example_subject), ("Class (Hz)", self.example_frequency),
                                   ("Block", self.example_block), ("Electrode", self.example_electrode),
                                   ("Encoder", self.example_encoder)):
                selector_row.addWidget(QLabel(label)); selector_row.addWidget(control)
            form.addRow("Selection", selectors); layout.addLayout(form)
            form.addRow("Preview", self.example_auto)
            self.example_generate_button = QPushButton("Generate signal/state/spike figure")
            self.example_generate_button.clicked.connect(self.generate_example)
            self.example_generate_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
            layout.addWidget(self.example_generate_button)
            self.example_progress = QProgressBar(); self._progress_idle(self.example_progress)
            layout.addWidget(self.example_progress)
            self.example_caption = QLabel("Choose a real segment to inspect raw and filtered EEG, internal state, threshold, and spikes.")
            layout.addWidget(self.example_caption)
            self.example_pixmap = None
            self.example_image = QLabel(); self.example_image.setAlignment(Qt.AlignCenter)
            self.example_image.setMinimumSize(320, 240)
            self.example_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.example_scroll = QScrollArea(); self.example_scroll.setWidgetResizable(True)
            self.example_scroll.setWidget(self.example_image); layout.addWidget(self.example_scroll, 1)
            self.example_timer = QTimer(self); self.example_timer.setSingleShot(True)
            self.example_timer.timeout.connect(self.generate_example)
            self.example_process.finished.connect(self.example_finished)
            for control in (self.example_subject, self.example_block):
                control.valueChanged.connect(self.schedule_example)
            for control in (self.example_frequency, self.example_electrode, self.example_encoder):
                control.currentIndexChanged.connect(self.schedule_example)
            self.tabs.addTab(tab, "Neuron explorer")

        def schedule_example(self, *_: object) -> None:
            if self.example_auto.isChecked():
                self.example_timer.start(350)

        def generate_example(self) -> None:
            try:
                encoder_map = {
                    "All encoders": ("resonate_fire", "delta", "lif"),
                    "Resonate-and-fire": ("resonate_fire",), "Delta": ("delta",), "LIF": ("lif",),
                }
                if self.example_process.state() != QProcess.NotRunning:
                    self.example_pending = True; self.example_caption.setText("Rendering current selection; latest change is queued..."); return
                subject = self.example_subject.value(); frequency = int(self.example_frequency.currentText())
                block = self.example_block.value(); electrode = self.example_electrode.currentText()
                output_dir = Path(self.example_output.text())
                stem = f"s{subject:02d}_{frequency:02d}hz_b{block:02d}_{electrode.lower()}"
                self.example_output_path = output_dir / f"{stem}.png"
                arguments = ["-m", "ssvep_toolkit.cli", "example-neuron", "--data-dir", self.example_data.text(),
                             "--output", str(self.example_output_path), "--subject", str(subject),
                             "--frequency", str(frequency), "--block", str(block), "--electrode", electrode,
                             "--encoders", *encoder_map[self.example_encoder.currentText()]]
                self.example_caption.setText("Rendering in the background...")
                self._progress_busy(self.example_progress, "Loading EEG and rendering full-resolution figure…")
                self.example_generate_button.setEnabled(False)
                self.example_process.start(__import__("sys").executable, arguments)
            except Exception as exc:
                QMessageBox.critical(self, "Example failed", str(exc))

        def example_finished(self, exit_code: int, *_: object) -> None:
            if exit_code == 0 and self.example_output_path.exists():
                self.example_pixmap = QPixmap(str(self.example_output_path)); self._fit_example_image()
                self.example_caption.setText(f"Created {self.example_output_path}")
            else:
                error = bytes(self.example_process.readAllStandardError()).decode(errors="replace").strip()
                self.example_caption.setText(f"Example failed: {error or 'unknown error'}")
            self.example_generate_button.setEnabled(True)
            self._progress_idle(self.example_progress, "Figure ready" if exit_code == 0 else "Figure failed")
            if self.example_pending:
                self.example_pending = False; self.example_timer.start(0)

        def example_process_error(self, *_: object) -> None:
            self.example_generate_button.setEnabled(True)
            self._progress_idle(self.example_progress, "Figure process failed")
            self.example_caption.setText(f"Figure process failed: {self.example_process.errorString()}")

        def _fit_example_image(self) -> None:
            if self.example_pixmap is None or self.example_pixmap.isNull():
                return
            available = self.example_scroll.viewport().size()
            fitted = self.example_pixmap.scaled(
                max(1, available.width() - 8), max(1, available.height() - 8),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.example_image.setPixmap(fitted)

        def resizeEvent(self, event: object) -> None:
            super().resizeEvent(event)
            QTimer.singleShot(0, self._fit_example_image)

        def _build_run_tab(self) -> None:
            tab = QWidget(); layout = QVBoxLayout(tab); form = QFormLayout()
            self.run_config = QLineEdit(str(Path("configs/nested_multi_encoder.yaml")))
            self.run_subject = self._spin(1, 30, 1); self.run_classes = QComboBox(); self.run_classes.addItems(("4", "16")); self.run_classes.setCurrentText("4")
            self.run_selection = QComboBox()
            self.run_selection.addItem("Manual start + spacing", "fixed_spacing_harmonic_aware")
            self.run_selection.addItem("Auto compact, fewest harmonic collisions", "compact_harmonic_aware")
            self.run_selection.addItem("Consecutive from lowest frequency (ablation)", "low_contiguous")
            self.run_selection.setToolTip("Manual uses f(k) = start + k x spacing. Auto compact searches consecutive frequencies. The ablation intentionally ignores harmonic overlap.")
            form.addRow("Experiment config", self._file_row(self.run_config, "YAML (*.yaml *.yml)"))
            selectors = QWidget(); row = QHBoxLayout(selectors); row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel("Subject")); row.addWidget(self.run_subject); row.addWidget(QLabel("Classes")); row.addWidget(self.run_classes)
            row.addWidget(QLabel("Class selection")); row.addWidget(self.run_selection); row.addStretch()
            form.addRow("Pilot cell", selectors); layout.addLayout(form)
            parameter_widget = QWidget(); parameters = QFormLayout(parameter_widget)
            self.run_start_hz = self._spin(1, 60, 17)
            self.run_spacing_hz = self._spin(1, 12, 4)
            self.run_duration_ms = self._spin(100, 5000, 500); self.run_duration_ms.setSingleStep(50)
            self.run_condition = self._spin(1, 2, 2)
            self.run_filter_modes = QComboBox(); self.run_filter_modes.addItems((
                "Both: offline benchmark + causal deployment",
                "Causal only: past samples only",
                "Offline only: zero-phase, uses future samples",
            ))
            self.run_filter_modes.setToolTip("Offline filtering is a non-causal upper-bound analysis. Causal filtering is deployable in real time and cannot use future EEG samples.")
            self.run_band_half_width = self._double(0.1, 10.0, 1.0, 2, 0.1)
            self.run_band_order = self._spin(1, 10, 5)
            self.run_rf_alpha = QLineEdit("0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4")
            self.run_rf_threshold = QLineEdit("0.00025, 0.001, 0.004, 0.016, 0.064, 0.256, 1.024")
            self.run_rf_operating = QLineEdit("0.75")
            self.run_rf_gain = self._double(0.0001, 10.0, 0.05, 4, 0.01)
            self.run_rf_compensate = QCheckBox("Divide input drive by resonance frequency")
            self.run_rf_compensate.setChecked(False)
            self.run_rf_harmonics = QLineEdit("1")
            self.run_rf_spread = QLineEdit("-0.5, 0, 0.5")
            self.run_rf_substeps = self._spin(1, 32, 4)
            self.run_rf_refractory = self._double(0.0, 5.0, 0.5, 2, 0.1)
            self.run_delta_threshold = QLineEdit("0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56")
            self.run_delta_asymmetry = QLineEdit("0.5, 0.75, 1.0, 1.3333333333, 2.0")
            self.run_lif_threshold = QLineEdit("0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16")
            self.run_lif_tau = QLineEdit("0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2")
            self.run_lif_gain = self._double(0.001, 20.0, 1.0, 3, 0.1)
            self.run_l2 = QLineEdit("0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000")
            self.run_opt_rule = QComboBox()
            self.run_opt_rule.addItem("One standard error (recommended)", "one_standard_error")
            self.run_opt_rule.addItem("Maximum inner mean (ablation)", "max_mean")
            self.run_opt_rule.setToolTip("Uses inner blocks only. One-SE avoids choosing a parameter for a one-trial validation fluctuation.")
            start_row = QWidget(); start_layout = QHBoxLayout(start_row); start_layout.setContentsMargins(0, 0, 0, 0)
            start_layout.addWidget(QLabel("Start")); start_layout.addWidget(self.run_start_hz); start_layout.addWidget(QLabel("Hz    Spacing")); start_layout.addWidget(self.run_spacing_hz); start_layout.addWidget(QLabel("Hz")); start_layout.addStretch()
            parameters.addRow("Stimulus grid", start_row)
            acquisition_row = QWidget(); acquisition_layout = QHBoxLayout(acquisition_row); acquisition_layout.setContentsMargins(0, 0, 0, 0)
            acquisition_layout.addWidget(QLabel("Decision")); acquisition_layout.addWidget(self.run_duration_ms); acquisition_layout.addWidget(QLabel("ms    Condition")); acquisition_layout.addWidget(self.run_condition); acquisition_layout.addWidget(QLabel("Filters")); acquisition_layout.addWidget(self.run_filter_modes); acquisition_layout.addStretch()
            parameters.addRow("Evaluation", acquisition_row)
            filter_row = QWidget(); filter_layout = QHBoxLayout(filter_row); filter_layout.setContentsMargins(0, 0, 0, 0)
            filter_layout.addWidget(QLabel("Half-width")); filter_layout.addWidget(self.run_band_half_width); filter_layout.addWidget(QLabel("Hz    Order")); filter_layout.addWidget(self.run_band_order); filter_layout.addStretch()
            parameters.addRow("Band-pass", filter_row)
            parameters.addRow("R&F damping alpha grid", self.run_rf_alpha)
            parameters.addRow("R&F threshold grid", self.run_rf_threshold)
            parameters.addRow("R&F operating RMS grid", self.run_rf_operating)
            rf_row = QWidget(); rf_layout = QHBoxLayout(rf_row); rf_layout.setContentsMargins(0, 0, 0, 0)
            rf_layout.addWidget(QLabel("Gain")); rf_layout.addWidget(self.run_rf_gain); rf_layout.addWidget(QLabel("Harmonics")); rf_layout.addWidget(self.run_rf_harmonics); rf_layout.addWidget(QLabel("Spread Hz")); rf_layout.addWidget(self.run_rf_spread); rf_layout.addWidget(QLabel("Substeps")); rf_layout.addWidget(self.run_rf_substeps); rf_layout.addWidget(QLabel("Refractory")); rf_layout.addWidget(self.run_rf_refractory)
            parameters.addRow("R&F bank", rf_row)
            parameters.addRow("R&F compatibility", self.run_rf_compensate)
            parameters.addRow("Delta threshold grid (uV)", self.run_delta_threshold)
            parameters.addRow("Delta asymmetry grid", self.run_delta_asymmetry)
            parameters.addRow("LIF threshold grid (uV)", self.run_lif_threshold)
            parameters.addRow("LIF tau grid (s)", self.run_lif_tau)
            parameters.addRow("LIF input gain", self.run_lif_gain)
            parameters.addRow("Fusion ridge L2 grid", self.run_l2)
            parameters.addRow("Selection rule", self.run_opt_rule)
            parameter_scroll = QScrollArea(); parameter_scroll.setWidgetResizable(True); parameter_scroll.setWidget(parameter_widget); parameter_scroll.setMaximumHeight(330)
            layout.addWidget(parameter_scroll)
            explanation = QLabel(
                "Class rule: class k = start + k x spacing. Example: start 17 Hz, spacing 4 Hz, four classes -> 17, 21, 25, 29 Hz. "
                "Offline filtering uses samples before and after each time point; causal filtering uses present/past samples only and is the real-time result. "
                "One-SE optimization treats statistically equivalent inner-fold settings as ties and uses the predeclared reference point."
            ); explanation.setWordWrap(True); layout.addWidget(explanation)
            self.run_cost_estimate = QLabel(); self.run_cost_estimate.setWordWrap(True)
            layout.addWidget(self.run_cost_estimate)
            self.run_frequency_plan = QLabel(); self.run_frequency_plan.setWordWrap(True)
            layout.addWidget(self.run_frequency_plan)
            self.run_classes.currentIndexChanged.connect(self.class_count_changed)
            self.run_selection.currentIndexChanged.connect(self.update_frequency_plan)
            self.run_start_hz.valueChanged.connect(self.update_frequency_plan)
            self.run_spacing_hz.valueChanged.connect(self.update_frequency_plan)
            for field in (self.run_rf_alpha, self.run_rf_threshold, self.run_rf_operating,
                          self.run_delta_threshold, self.run_delta_asymmetry,
                          self.run_lif_threshold, self.run_lif_tau, self.run_l2):
                field.textChanged.connect(self.update_cost_estimate)
            self.run_filter_modes.currentIndexChanged.connect(self.update_cost_estimate)
            self.update_cost_estimate()
            self.update_frequency_plan()
            self.existing_experiment_status = QLabel(); self.existing_experiment_status.setWordWrap(True)
            layout.addWidget(self.existing_experiment_status)
            self.recent_experiments = QTableWidget(3, 4); self.recent_experiments.setHorizontalHeaderLabels(("Recent experiment", "Checkpoints", "State", "Last updated"))
            self.recent_experiments.verticalHeader().setVisible(False); self.recent_experiments.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.recent_experiments.setMaximumHeight(126); self.recent_experiments.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(self.recent_experiments)
            buttons = QHBoxLayout(); self.run_start_button = QPushButton("Run nested pilot"); self.run_stop_button = QPushButton("Stop"); self.refresh_button = QPushButton("Refresh existing jobs"); self.cleanup_button = QPushButton("Clean old GUI runs")
            self.run_start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay)); self.run_stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
            self.refresh_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload)); self.cleanup_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
            self.cleanup_keep = self._spin(1, 20, 3); self.cleanup_keep.setToolTip("Keep at least this many newest GUI runs. Only runs older than 30 days are eligible.")
            self.run_start_button.clicked.connect(self.start_run); self.run_stop_button.clicked.connect(self.process.terminate)
            self.refresh_button.clicked.connect(self.refresh_experiment_status)
            self.cleanup_button.clicked.connect(self.cleanup_old_gui_runs)
            self.run_stop_button.setEnabled(False)
            buttons.addWidget(self.run_start_button); buttons.addWidget(self.run_stop_button); buttons.addWidget(self.refresh_button); buttons.addWidget(self.cleanup_button); buttons.addWidget(QLabel("Keep")); buttons.addWidget(self.cleanup_keep); buttons.addStretch(); layout.addLayout(buttons)
            self.run_progress = QProgressBar(); self._progress_idle(self.run_progress, "Experiment ready")
            self.maintenance_progress = QProgressBar(); self._progress_idle(self.maintenance_progress, "Storage scan ready")
            layout.addWidget(self.run_progress); layout.addWidget(self.maintenance_progress)
            self.run_log = QTextEdit(); self.run_log.setReadOnly(True); layout.addWidget(self.run_log)
            self.process.readyReadStandardOutput.connect(self.read_output); self.process.readyReadStandardError.connect(self.read_error)
            self.process.finished.connect(self.experiment_finished)
            QTimer.singleShot(0, self.refresh_experiment_status)
            self.tabs.addTab(tab, "Experiments")

        def start_run(self) -> None:
            if self.process.state() != QProcess.NotRunning:
                QMessageBox.information(self, "Experiment running", "Wait for or stop the current process."); return
            try:
                from datetime import datetime
                import yaml

                config = yaml.safe_load(Path(self.run_config.text()).read_text(encoding="utf-8"))
                count = int(self.run_classes.currentText())
                from ssvep_toolkit.evaluation import select_class_frequencies
                fixed = self.run_selection.currentData() == "fixed_spacing_harmonic_aware"
                select_class_frequencies(
                    count, available_hz=(1, 60), strategy=str(self.run_selection.currentData()),
                    spacing_hz=self.run_spacing_hz.value() if fixed else None,
                    start_hz=self.run_start_hz.value() if fixed else None,
                )
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_s{self.run_subject.value():02d}_{count:02d}c"
                study = config["study"]
                study.update(
                    name=f"gui-{run_id}", subjects=[self.run_subject.value()], class_counts=[count],
                    condition=self.run_condition.value(), decision_seconds=self.run_duration_ms.value() / 1000.0,
                    filter_modes={
                        "Both: offline benchmark + causal deployment": ["offline", "causal"],
                        "Causal only: past samples only": ["causal"],
                        "Offline only: zero-phase, uses future samples": ["offline"],
                    }[self.run_filter_modes.currentText()],
                    output=f"outputs/experiments/gui_runs/{run_id}",
                )
                study["class_selection"] = {
                    "strategy": str(self.run_selection.currentData()), "available_hz": [1, 60],
                    "interference_harmonics": [2, 3], "start_hz": self.run_start_hz.value(),
                    "spacing_hz_by_class_count": {count: self.run_spacing_hz.value()},
                }
                config["bandpass"].update(order=self.run_band_order.value(), half_width_hz=self.run_band_half_width.value())
                config["resonate_fire"].update(
                    damping_alpha=self._number_grid(self.run_rf_alpha.text()),
                    threshold=self._number_grid(self.run_rf_threshold.text()),
                    operating_rms=self._number_grid(self.run_rf_operating.text()),
                    input_gain=self.run_rf_gain.value(), harmonics=self._integer_grid(self.run_rf_harmonics.text()),
                    spread_hz=self._number_grid(self.run_rf_spread.text()),
                    integration_substeps=self.run_rf_substeps.value(), refractory_cycles=self.run_rf_refractory.value(),
                    normalize_input_by_resonance=self.run_rf_compensate.isChecked(),
                )
                config["delta"].update(
                    threshold_uV=self._number_grid(self.run_delta_threshold.text()),
                    asymmetry=self._number_grid(self.run_delta_asymmetry.text()),
                )
                config["lif"].update(
                    threshold_uV=self._number_grid(self.run_lif_threshold.text()),
                    tau_seconds=self._number_grid(self.run_lif_tau.text()), input_gain=self.run_lif_gain.value(),
                )
                config["fusion"]["l2_grid"] = self._number_grid(self.run_l2.text())
                config.setdefault("optimization", {}).update(
                    candidate_selection_rule=str(self.run_opt_rule.currentData()),
                    l2_selection_rule=str(self.run_opt_rule.currentData()),
                )
                config_path = Path("outputs/gui_configs") / f"{run_id}.yaml"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                self.current_run_log_path = config_path.with_suffix(".log")
                self.run_output_buffer = ""
                script = Path.cwd() / "scripts/run_nested_multi_encoder.py"
                arguments = [str(script), "--config", str(config_path), "--subjects", str(self.run_subject.value()),
                             "--class-counts", str(count), "--class-selection-strategy", str(self.run_selection.currentData())]
                self.run_log.append(f"Saved configuration: {config_path}")
                self.run_log.append("Starting: python " + " ".join(arguments))
                self._progress_busy(self.run_progress, "Starting experiment process…")
                self.run_start_button.setEnabled(False); self.run_stop_button.setEnabled(True)
                self.process.start(__import__("sys").executable, arguments)
            except Exception as exc:
                QMessageBox.critical(self, "Experiment configuration failed", str(exc))

        @staticmethod
        def _number_grid(text: str) -> list[float]:
            values = [float(item.strip()) for item in text.split(",") if item.strip()]
            if not values:
                raise ValueError("parameter grids cannot be empty")
            return values

        @classmethod
        def _integer_grid(cls, text: str) -> list[int]:
            values = cls._number_grid(text)
            if any(not value.is_integer() for value in values):
                raise ValueError("harmonics must be integers")
            return [int(value) for value in values]

        def class_count_changed(self, *_: object) -> None:
            self.run_spacing_hz.setValue({4: 4, 16: 2}[int(self.run_classes.currentText())])
            self.update_frequency_plan()

        @staticmethod
        def _scan_experiments() -> dict[str, object]:
            from datetime import datetime
            root = Path("outputs/experiments")
            def count(relative: str) -> int:
                directory = root / relative / "checkpoints"
                return sum(1 for _ in directory.glob("*.npz")) if directory.exists() else 0
            counts = {
                "endpoint": count("resonate_and_fire_decision_endpoints"),
                "fused": count("resonate_and_fire_fused_reference_search"),
                "delta_lif": count("individual_spike_encoders"),
                "revised": count("nested_multi_encoder_4c16c_confirmatory_v2"),
                "gui": sum(1 for _ in (root / "gui_runs").glob("*/checkpoints/*.npz"))
                if (root / "gui_runs").exists() else 0,
            }
            planned = {
                "resonate_and_fire_decision_endpoints": 450,
                "resonate_and_fire_fused_reference_search": 150,
                "individual_spike_encoders": 300,
                "nested_multi_encoder_harmonic_aware": 120,
                "nested_multi_encoder_4c16c_confirmatory_v2": 120,
            }
            directories = [path for path in root.iterdir() if path.is_dir() and path.name != "gui_runs"] if root.exists() else []
            gui_root = root / "gui_runs"
            if gui_root.exists(): directories.extend(path for path in gui_root.iterdir() if path.is_dir())
            candidates = []
            for directory in directories:
                updated = directory.stat().st_mtime
                for path in directory.rglob("*"):
                    if path.is_file():
                        updated = max(updated, path.stat().st_mtime)
                checkpoints = sum(1 for _ in (directory / "checkpoints").glob("*.npz"))
                expected = planned.get(directory.name)
                state = f"{100 * checkpoints / expected:.1f}%" if expected else ("has checkpoints" if checkpoints else "configured/no checkpoint")
                candidates.append((updated, directory.name, checkpoints, state))
            recent = [
                (name, str(checkpoints), state, datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M:%S"))
                for updated, name, checkpoints, state in sorted(candidates, reverse=True)[:3]
            ]
            return {"counts": counts, "recent": recent}

        def refresh_experiment_status(self) -> None:
            if self.refresh_active or self.cleanup_active:
                return
            self.refresh_active = True; self.refresh_button.setEnabled(False); self.cleanup_button.setEnabled(False)
            self._progress_busy(self.maintenance_progress, "Scanning experiment checkpoints and storage…")
            def result(snapshot: dict[str, object]) -> None:
                counts = snapshot["counts"]
                self.existing_experiment_status.setText(
                    f"Existing checkpoints — endpoints {counts['endpoint']}/450; fused R&F {counts['fused']}/150; "
                    f"delta/LIF {counts['delta_lif']}/300; revised 4/16-class study {counts['revised']}/120; GUI runs {counts['gui']}."
                )
                recent = snapshot["recent"]; self.recent_experiments.setRowCount(3)
                for row in range(3):
                    values = recent[row] if row < len(recent) else ("", "", "", "")
                    for column, value in enumerate(values):
                        self.recent_experiments.setItem(row, column, QTableWidgetItem(value))
                self.recent_experiments.resizeColumnsToContents()
            def error(message: str) -> None:
                self.existing_experiment_status.setText(f"Could not read recent experiments: {message}")
            def finished() -> None:
                self.refresh_active = False; self.refresh_button.setEnabled(True); self.cleanup_button.setEnabled(True)
                self._progress_idle(self.maintenance_progress, "Storage scan complete")
            self._run_worker(self._scan_experiments, result, error, finished)

        def refresh_recent_experiments(self) -> None:
            self.refresh_experiment_status()

        def update_frequency_plan(self, *_: object) -> None:
            from ssvep_toolkit.evaluation import harmonic_collisions, select_class_frequencies
            try:
                fixed = self.run_selection.currentData() == "fixed_spacing_harmonic_aware"
                frequencies = select_class_frequencies(
                    int(self.run_classes.currentText()), available_hz=(1, 60),
                    strategy=str(self.run_selection.currentData()),
                    spacing_hz=self.run_spacing_hz.value() if fixed else None,
                    start_hz=self.run_start_hz.value() if fixed else None,
                )
                collisions = harmonic_collisions(frequencies)
                collision_text = "none" if not collisions else ", ".join(
                    f"{item.source_hz}x{item.harmonic}={item.target_hz} Hz" for item in collisions
                )
                self.run_frequency_plan.setText(
                    f"Stimulus classes: {', '.join(map(str, frequencies))} Hz. Exact 2x/3x collisions: {collision_text}."
                )
            except ValueError as exc:
                self.run_frequency_plan.setText(f"Invalid stimulus grid: {exc}")

        def update_cost_estimate(self, *_: object) -> None:
            try:
                rf = len(self._number_grid(self.run_rf_alpha.text())) * len(self._number_grid(self.run_rf_threshold.text())) * len(self._number_grid(self.run_rf_operating.text()))
                delta = len(self._number_grid(self.run_delta_threshold.text())) * len(self._number_grid(self.run_delta_asymmetry.text()))
                lif = len(self._number_grid(self.run_lif_threshold.text())) * len(self._number_grid(self.run_lif_tau.text()))
                ridge = len(self._number_grid(self.run_l2.text()))
                modes = 2 if self.run_filter_modes.currentIndex() == 0 else 1
                self.run_cost_estimate.setText(
                    f"Planned workload for this pilot: {modes} result cell(s); {rf} R&F, {delta} delta, "
                    f"{lif} LIF, and {ridge} ridge candidates. R&F retains rate, TTFS, cosine phase, and sine phase. "
                    "Multi-fidelity pruning uses 4/8/all inner blocks; eight branch/encoder ablations reuse fitted outer models. "
                    "The exact plan, configuration hash, dataset manifest, phase timing, and ETA are written beside the checkpoints."
                )
            except ValueError as exc:
                self.run_cost_estimate.setText(f"Workload estimate unavailable: {exc}")

        def read_output(self) -> None:
            self._append_run_output(bytes(self.process.readAllStandardOutput()).decode(errors="replace"))

        def read_error(self) -> None:
            self._append_run_output(bytes(self.process.readAllStandardError()).decode(errors="replace"))

        def _append_run_output(self, message: str) -> None:
            if not message:
                return
            if self.current_run_log_path is not None:
                with self.current_run_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(message)
            self.run_output_buffer += message
            while "\n" in self.run_output_buffer:
                line, self.run_output_buffer = self.run_output_buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("@@PROGRESS|"):
                    parts = line.split("|", 4)
                    if len(parts) == 5:
                        current, total = int(parts[2]), max(1, int(parts[3]))
                        self.run_progress.setRange(0, total); self.run_progress.setValue(current)
                        self.run_progress.setFormat(f"{parts[4]} — {current}/{total} (%p%)")
                elif line.startswith("@@BUSY|"):
                    self._progress_busy(self.run_progress, line.split("|", 1)[1] + "…")
                elif line:
                    self.run_log.append(line)

        def experiment_finished(self, exit_code: int, *_: object) -> None:
            if self.run_output_buffer.strip():
                self.run_log.append(self.run_output_buffer.strip())
            self.run_output_buffer = ""
            state = "completed" if exit_code == 0 else f"stopped or failed (exit {exit_code})"
            self.run_log.append(f"Experiment {state}. Persistent log: {self.current_run_log_path}")
            self.run_start_button.setEnabled(True); self.run_stop_button.setEnabled(False)
            self._progress_idle(self.run_progress, "Experiment complete" if exit_code == 0 else "Experiment stopped or failed")
            self.refresh_experiment_status()

        def experiment_process_error(self, *_: object) -> None:
            self.run_start_button.setEnabled(True); self.run_stop_button.setEnabled(False)
            self._progress_idle(self.run_progress, "Experiment process failed")
            self.run_log.append(f"Experiment process error: {self.process.errorString()}")

        def cleanup_old_gui_runs(self) -> None:
            if self.refresh_active or self.cleanup_active:
                return
            from ssvep_toolkit.storage import old_gui_run_candidates
            keep = self.cleanup_keep.value(); root = Path("outputs/experiments/gui_runs")
            self.cleanup_active = True; self.cleanup_stage = "scanning"
            self.refresh_button.setEnabled(False); self.cleanup_button.setEnabled(False)
            self._progress_busy(self.maintenance_progress, "Finding eligible old GUI runs and stale partial files…")
            def scan() -> tuple[tuple[Path, ...], tuple[Path, ...]]:
                return (
                    old_gui_run_candidates(root, keep=keep, older_than_days=30),
                    tuple(Path("outputs").rglob("*.partial.npz")),
                )
            def result(payload: tuple[tuple[Path, ...], tuple[Path, ...]]) -> None:
                candidates, partials = payload
                if not candidates and not partials:
                    QMessageBox.information(self, "Nothing to clean", "No GUI runs older than 30 days and no temporary partial files were found.")
                    return
                detail = "\n".join(path.name for path in candidates[:10]) or "No old GUI run directories"
                answer = QMessageBox.question(
                    self, "Confirm conservative cleanup",
                    f"Delete {len(candidates)} GUI run directories older than 30 days while keeping the newest {keep}?\n\n{detail}\n\nStale partial files older than 24 hours will also be removed. Completed legacy scientific experiments are untouched.",
                )
                if answer == QMessageBox.Yes:
                    self.cleanup_stage = "deleting"; self._begin_cleanup_delete(root, keep)
            def error(message: str) -> None:
                QMessageBox.critical(self, "Cleanup scan failed", message)
            def finished() -> None:
                if self.cleanup_stage == "scanning":
                    self.cleanup_active = False; self.refresh_button.setEnabled(True); self.cleanup_button.setEnabled(True)
                    self._progress_idle(self.maintenance_progress, "Cleanup scan complete")
            self._run_worker(scan, result, error, finished)

        def _begin_cleanup_delete(self, root: Path, keep: int) -> None:
            from ssvep_toolkit.storage import prune_old_gui_runs, remove_stale_partial_files
            self._progress_busy(self.maintenance_progress, "Removing confirmed old GUI data…")
            def remove() -> tuple[int, int]:
                removed_runs = prune_old_gui_runs(root, keep=keep, older_than_days=30)
                removed_partials = remove_stale_partial_files("outputs", older_than_hours=24)
                return len(removed_runs), len(removed_partials)
            def result(counts: tuple[int, int]) -> None:
                QMessageBox.information(self, "Cleanup complete", f"Removed {counts[0]} old GUI runs and {counts[1]} stale partial files. This deletion is not recoverable.")
            def error(message: str) -> None:
                QMessageBox.critical(self, "Cleanup failed", message)
            def finished() -> None:
                self.cleanup_active = False; self.cleanup_stage = "idle"
                self.refresh_button.setEnabled(True); self.cleanup_button.setEnabled(True)
                self._progress_idle(self.maintenance_progress, "Cleanup complete")
                self.refresh_experiment_status()
            self._run_worker(remove, result, error, finished)

        def _build_dashboard_tab(self) -> None:
            tab = QWidget(); layout = QVBoxLayout(tab); form = QFormLayout()
            self.registry_path = QLineEdit("outputs/registry/experiments.sqlite3")
            self.dashboard_path = QLineEdit("outputs/dashboard/index.html")
            self.dashboard_examples = QLineEdit("outputs/examples/neuron_behavior")
            form.addRow("Registry", self._file_row(self.registry_path, "SQLite (*.sqlite *.sqlite3)"))
            form.addRow("Dashboard output", self.dashboard_path)
            form.addRow("Neuron examples", self._folder_row(self.dashboard_examples)); layout.addLayout(form)
            buttons = QHBoxLayout(); self.dashboard_build_button = QPushButton("Build dashboard"); self.dashboard_open_button = QPushButton("Open dashboard")
            self.dashboard_build_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton)); self.dashboard_open_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
            self.dashboard_build_button.clicked.connect(self.build_dashboard); self.dashboard_open_button.clicked.connect(self.open_dashboard)
            buttons.addWidget(self.dashboard_build_button); buttons.addWidget(self.dashboard_open_button); buttons.addStretch(); layout.addLayout(buttons)
            self.dashboard_status = QLabel("The dashboard indexes registered runs, metrics, artifacts, and generated neuron examples.")
            self.dashboard_progress = QProgressBar(); self._progress_idle(self.dashboard_progress, "Dashboard ready")
            layout.addWidget(self.dashboard_progress); layout.addWidget(self.dashboard_status); layout.addStretch(); self.tabs.addTab(tab, "Dashboard")
            self.dashboard_process.finished.connect(self.dashboard_finished)

        def build_dashboard(self, open_after: bool = False) -> None:
            if self.dashboard_process.state() != QProcess.NotRunning:
                self.dashboard_open_after = self.dashboard_open_after or open_after
                return
            self.dashboard_open_after = bool(open_after)
            arguments = ["-m", "ssvep_toolkit.cli", "dashboard",
                         "--database", self.registry_path.text(), "--output", self.dashboard_path.text(),
                         "--examples", self.dashboard_examples.text()]
            self._progress_busy(self.dashboard_progress, "Indexing runs, metrics, examples, and storage…")
            self.dashboard_status.setText("Building dashboard in a separate process; the GUI remains responsive.")
            self.dashboard_build_button.setEnabled(False); self.dashboard_open_button.setEnabled(False)
            self.dashboard_process.start(__import__("sys").executable, arguments)

        def dashboard_finished(self, exit_code: int, *_: object) -> None:
            output = bytes(self.dashboard_process.readAllStandardOutput()).decode(errors="replace").strip()
            error = bytes(self.dashboard_process.readAllStandardError()).decode(errors="replace").strip()
            self.dashboard_build_button.setEnabled(True); self.dashboard_open_button.setEnabled(True)
            if exit_code == 0:
                self.dashboard_status.setText(output or f"Created {Path(self.dashboard_path.text()).resolve()}")
                self._progress_idle(self.dashboard_progress, "Dashboard complete")
                if self.dashboard_open_after:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.dashboard_path.text()).resolve())))
            else:
                self.dashboard_status.setText(error or f"Dashboard failed with exit code {exit_code}")
                self._progress_idle(self.dashboard_progress, "Dashboard failed")
            self.dashboard_open_after = False

        def dashboard_process_error(self, *_: object) -> None:
            self.dashboard_build_button.setEnabled(True); self.dashboard_open_button.setEnabled(True)
            self._progress_idle(self.dashboard_progress, "Dashboard process failed")
            self.dashboard_status.setText(f"Dashboard process failed: {self.dashboard_process.errorString()}")

        def open_dashboard(self) -> None:
            path = Path(self.dashboard_path.text()).resolve()
            if not path.exists():
                self.build_dashboard(open_after=True)
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    app = QApplication.instance() or QApplication([])
    window = Window(); window.show(); return app.exec()
