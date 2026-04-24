#!/usr/bin/env python3
"""PatcherBot evaluation GUI backed by shared robomimic utilities."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from .config import Config
    from .evaluation_manager import EvaluationManager
    from .results_viewer import ResultsViewerWidget
except ImportError:
    from config import Config
    from evaluation_manager import EvaluationManager
    from results_viewer import ResultsViewerWidget


class EvaluationWorker(QThread):
    """Runs checkpoint evaluation and plot generation off the UI thread."""

    progress = pyqtSignal(str, int)
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, manager: EvaluationManager, checkpoint_dir: str):
        super().__init__()
        self.manager = manager
        self.checkpoint_dir = checkpoint_dir

    def run(self):
        try:
            results = self.manager.evaluate_all_checkpoints(self.checkpoint_dir)
            total = max(len(results), 1)
            for index, result in enumerate(results, start=1):
                checkpoint_name = Path(result.get("checkpoint", "unknown")).name
                if result.get("success"):
                    self.manager.enrich_result_with_plots(result)
                    message = f"Processed {checkpoint_name}"
                else:
                    message = f"Failed {checkpoint_name}"
                self.progress.emit(message, int(index / total * 100))
            self.result.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class PatcherBotGUI(QMainWindow):
    """Main GUI window for PatcherBot evaluation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatcherBot Evaluation GUI")
        self.setMinimumSize(1280, 840)

        self.config = Config()
        self.config.sync_output_dirs()
        self.evaluation_manager = EvaluationManager(self.config)
        self.viewer = ResultsViewerWidget(self)
        self.worker: Optional[EvaluationWorker] = None
        self.results: List[Dict[str, Any]] = []

        self._create_widgets()
        self._setup_connections()

    def _create_widgets(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        main_layout.addWidget(self._create_toolbar())

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._create_tab_widget())
        splitter.addWidget(self._create_config_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.statusBar().showMessage("Ready")

    def _create_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_evaluate = QPushButton("Evaluate Checkpoints")
        self.btn_select_csv = QPushButton("Open CSV")
        self.btn_select_dir = QPushButton("Select Checkpoint Dir")
        self.btn_reset = QPushButton("Reset")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        layout.addWidget(self.btn_evaluate)
        layout.addWidget(self.btn_select_csv)
        layout.addWidget(self.btn_select_dir)
        layout.addWidget(self.progress_bar, 1)
        layout.addWidget(self.btn_reset)
        return toolbar

    def _create_tab_widget(self) -> QTabWidget:
        tab_widget = QTabWidget()

        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)
        self.plot_content_layout = QVBoxLayout(scroll_content)

        self.plot_placeholder = QLabel("Select a CSV file or evaluate checkpoints to view plots.")
        self.plot_placeholder.setAlignment(Qt.AlignCenter)
        self.plot_placeholder.setWordWrap(True)
        self.plot_placeholder.setStyleSheet("background-color: #f0f0f0; padding: 20px;")

        self.plot_title = QLabel("")
        self.plot_title.setWordWrap(True)
        self.plot_title.hide()

        self.pred_plot_label = QLabel()
        self.pred_plot_label.setAlignment(Qt.AlignCenter)
        self.pred_plot_label.setWordWrap(True)
        self.pred_plot_label.hide()

        self.fft_title = QLabel("FFT")
        self.fft_title.hide()

        self.fft_plot_label = QLabel()
        self.fft_plot_label.setAlignment(Qt.AlignCenter)
        self.fft_plot_label.setWordWrap(True)
        self.fft_plot_label.hide()

        self.plot_content_layout.addWidget(self.plot_placeholder)
        self.plot_content_layout.addWidget(self.plot_title)
        self.plot_content_layout.addWidget(self.pred_plot_label)
        self.plot_content_layout.addWidget(self.fft_title)
        self.plot_content_layout.addWidget(self.fft_plot_label)
        self.plot_content_layout.addStretch()
        plots_layout.addWidget(scroll_area)
        tab_widget.addTab(plots_widget, "Plots")

        metadata_widget = QWidget()
        metadata_layout = QVBoxLayout(metadata_widget)
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setPlaceholderText("Metadata will appear here after evaluation...")
        metadata_layout.addWidget(self.metadata_text)
        tab_widget.addTab(metadata_widget, "Metadata")

        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(11)
        self.summary_table.setHorizontalHeaderLabels(
            [
                "Checkpoint",
                "Demo Index",
                "Demo Key",
                "Success",
                "Steps",
                "Mean L2",
                "Median L2",
                "Mean Reward",
                "CSV Path",
                "Metadata Path",
                "Error",
            ]
        )
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(8, QHeaderView.Interactive)
        self.summary_table.horizontalHeader().setSectionResizeMode(9, QHeaderView.Interactive)
        self.summary_table.horizontalHeader().setSectionResizeMode(10, QHeaderView.Stretch)
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.summary_table.setSelectionMode(QTableWidget.SingleSelection)
        self.summary_table.setAlternatingRowColors(True)
        summary_layout.addWidget(self.summary_table)
        tab_widget.addTab(summary_widget, "Summary")

        return tab_widget

    def _create_config_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        group = QGroupBox("Configuration")
        form_layout = QFormLayout(group)

        self.input_dataset = QLineEdit(self.config.dataset_path)
        self.input_dataset.setPlaceholderText("Path to test dataset (.hdf5)")
        form_layout.addRow("Dataset:", self.input_dataset)

        self.input_checkpoint_dir = QLineEdit(self.config.checkpoint_dir)
        self.input_checkpoint_dir.setPlaceholderText("Path to checkpoint directory")
        form_layout.addRow("Checkpoint Directory:", self.input_checkpoint_dir)

        self.input_csv_dir = QLineEdit(self.config.csv_dir)
        self.input_csv_dir.setPlaceholderText("Directory for per-checkpoint CSV outputs")
        form_layout.addRow("CSV Output Dir:", self.input_csv_dir)

        self.input_metadata_dir = QLineEdit(self.config.metadata_dir)
        self.input_metadata_dir.setPlaceholderText("Directory for per-checkpoint metadata JSON outputs")
        form_layout.addRow("Metadata Dir:", self.input_metadata_dir)

        self.spin_horizon = QSpinBox()
        self.spin_horizon.setMinimum(0)
        self.spin_horizon.setMaximum(100000)
        self.spin_horizon.setSpecialValueText("Auto")
        self.spin_horizon.setValue(0)
        form_layout.addRow("Horizon:", self.spin_horizon)

        self.spin_eps = QDoubleSpinBox()
        self.spin_eps.setMinimum(-10.0)
        self.spin_eps.setMaximum(1.0)
        self.spin_eps.setSingleStep(0.1)
        self.spin_eps.setValue(self.config.eps)
        form_layout.addRow("Success Epsilon:", self.spin_eps)

        self.chk_show_pos_traj = QCheckBox("Compute positional trajectory error")
        self.chk_show_pos_traj.setChecked(self.config.show_pos_traj)
        form_layout.addRow(self.chk_show_pos_traj)

        layout.addWidget(group)

        info = QLabel(
            "The GUI writes per-checkpoint CSV and metadata files, then renders plots using the shared robomimic plotting utilities."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)
        layout.addStretch()

        return panel

    def _setup_connections(self):
        self.btn_evaluate.clicked.connect(self._on_evaluate)
        self.btn_select_csv.clicked.connect(self._on_select_csv)
        self.btn_select_dir.clicked.connect(self._on_select_dir)
        self.btn_reset.clicked.connect(self._on_reset)
        self.summary_table.itemSelectionChanged.connect(self._on_summary_selection_changed)

        self.input_dataset.textChanged.connect(self._on_config_changed)
        self.input_checkpoint_dir.textChanged.connect(self._on_config_changed)
        self.input_csv_dir.textChanged.connect(self._on_config_changed)
        self.input_metadata_dir.textChanged.connect(self._on_config_changed)
        self.spin_horizon.valueChanged.connect(self._on_config_changed)
        self.spin_eps.valueChanged.connect(self._on_config_changed)
        self.chk_show_pos_traj.toggled.connect(self._on_config_changed)

    def _on_config_changed(self):
        self.config.dataset_path = self.input_dataset.text().strip()
        self.config.checkpoint_dir = self.input_checkpoint_dir.text().strip()
        self.config.csv_dir = self.input_csv_dir.text().strip() or self.config.DEFAULT_CSV_DIR
        self.config.metadata_dir = self.input_metadata_dir.text().strip() or self.config.csv_dir
        self.config.horizon = self.spin_horizon.value() or None
        self.config.eps = self.spin_eps.value()
        self.config.show_pos_traj = self.chk_show_pos_traj.isChecked()
        self.config.plots_dir = str(self.config.resolve_csv_dir() / "plots")
        self.config.evaluate_all_demos = True
        self.config.demo_id = None
        self.config.sync_output_dirs()

    def _on_evaluate(self):
        self._on_config_changed()
        checkpoint_dir = self.config.checkpoint_dir
        if not Path(checkpoint_dir).exists():
            QMessageBox.warning(self, "Error", f"Checkpoint directory not found:\n{checkpoint_dir}")
            return

        pth_files = sorted(Path(checkpoint_dir).glob("*.pth"))
        if not pth_files:
            QMessageBox.warning(
                self,
                "No Checkpoints",
                f"No .pth checkpoints found in:\n{checkpoint_dir}\n\nPlease select a directory containing .pth files.",
            )
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_evaluate.setEnabled(False)
        self.statusBar().showMessage("Evaluating checkpoints...")

        self.worker = EvaluationWorker(self.evaluation_manager, checkpoint_dir)
        self.worker.progress.connect(self._on_evaluation_progress)
        self.worker.result.connect(self._on_evaluation_complete)
        self.worker.error.connect(self._on_evaluation_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_finished(self):
        self.btn_evaluate.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.worker = None

    def _on_evaluation_progress(self, message: str, value: int):
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)

    def _on_evaluation_complete(self, results: List[Dict[str, Any]]):
        self.results = list(results)
        self.viewer.set_results(results)
        self._update_summary_table(results)

        first_success = next((idx for idx, result in enumerate(results) if result.get("success")), None)
        row_to_select = first_success if first_success is not None else (0 if results else None)
        if row_to_select is not None:
            self.summary_table.selectRow(row_to_select)
            self._display_result(results[row_to_select])
        else:
            self._clear_details()

        success_count = sum(1 for result in results if result.get("success"))
        self.statusBar().showMessage(f"Evaluation complete: {success_count}/{len(results)} checkpoints succeeded")

    def _on_evaluation_error(self, error: str):
        self.btn_evaluate.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Evaluation Error", f"An error occurred during evaluation:\n\n{error}")

    def _on_select_csv(self):
        csv_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if csv_path:
            self._load_csv(csv_path)

    def _on_select_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Checkpoint Directory", self.config.checkpoint_dir)
        if dir_path:
            self.input_checkpoint_dir.setText(dir_path)

    def _load_csv(self, csv_path: str):
        self._on_config_changed()
        try:
            plot_paths = self.evaluation_manager.generate_plot_bundle(csv_path)
        except Exception as exc:
            QMessageBox.critical(self, "Plot Error", f"Failed to render CSV plots:\n\n{exc}")
            return

        result = {
            "success": True,
            "checkpoint": Path(csv_path).name,
            "csv_path": str(Path(csv_path).expanduser()),
            "metadata_path": "",
            "summary": {},
            "plot_paths": plot_paths,
        }
        self.results = [result]
        self.viewer.set_results(self.results)
        self._update_summary_table(self.results)
        self.summary_table.selectRow(0)
        self._display_result(result)
        self.statusBar().showMessage(f"Loaded {Path(csv_path).name}")

    def _on_summary_selection_changed(self):
        selected_rows = self.summary_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self.results):
            self._display_result(self.results[row])

    def _display_result(self, result: Dict[str, Any]):
        self.metadata_text.setText(self.viewer.metadata_text(result))

        plot_paths = result.get("plot_paths") or {}
        main_plot = plot_paths.get("main")
        fft_plot = plot_paths.get("fft")
        if not main_plot:
            self._clear_plots_only()
            self.plot_placeholder.setText("No plots are available for the selected result.")
            self.plot_placeholder.show()
            return

        self.plot_placeholder.hide()
        self.plot_title.setText(self.viewer.display_name(result))
        self.plot_title.show()
        self._set_plot_pixmap(self.pred_plot_label, main_plot)

        if fft_plot and Path(fft_plot).exists():
            self.fft_title.show()
            self._set_plot_pixmap(self.fft_plot_label, fft_plot)
        else:
            self.fft_title.hide()
            self.fft_plot_label.hide()

    def _set_plot_pixmap(self, label: QLabel, image_path: str):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            label.setText(f"Could not load image:\n{image_path}")
            label.setPixmap(QPixmap())
        else:
            label.setText("")
            label.setPixmap(pixmap.scaledToWidth(1100, Qt.SmoothTransformation))
        label.show()

    def _update_summary_table(self, results: List[Dict[str, Any]]):
        self.summary_table.setRowCount(len(results))

        for row_idx, result in enumerate(results):
            checkpoint_name = Path(result.get("checkpoint", "")).name or "Unknown"
            summary = result.get("summary") or {}

            self.summary_table.setItem(row_idx, 0, QTableWidgetItem(checkpoint_name))

            demo_id = result.get("demo_id")
            demo_key = result.get("demo_key", "")
            self.summary_table.setItem(row_idx, 1, QTableWidgetItem("" if demo_id is None else str(demo_id)))
            self.summary_table.setItem(row_idx, 2, QTableWidgetItem("" if demo_key is None else str(demo_key)))

            success_item = QTableWidgetItem("Yes" if result.get("success") else "No")
            success_item.setForeground(QColor("darkgreen") if result.get("success") else QColor("darkred"))
            self.summary_table.setItem(row_idx, 3, success_item)

            steps = summary.get("steps")
            self.summary_table.setItem(row_idx, 4, QTableWidgetItem(str(steps) if steps is not None else ""))

            mean_l2 = summary.get("mean_l2")
            self.summary_table.setItem(row_idx, 5, QTableWidgetItem(f"{mean_l2:.6f}" if mean_l2 is not None else ""))

            median_l2 = summary.get("l2_metrics", {}).get("median") if summary.get("l2_metrics") else None
            self.summary_table.setItem(row_idx, 6, QTableWidgetItem(f"{median_l2:.6f}" if median_l2 is not None else ""))

            mean_reward = summary.get("mean_reward")
            self.summary_table.setItem(row_idx, 7, QTableWidgetItem(f"{mean_reward:.6f}" if mean_reward is not None else ""))

            self.summary_table.setItem(row_idx, 8, QTableWidgetItem(result.get("csv_path", "")))
            self.summary_table.setItem(row_idx, 9, QTableWidgetItem(result.get("metadata_path", "")))
            self.summary_table.setItem(row_idx, 10, QTableWidgetItem(result.get("error", "")))

        if results:
            self.summary_table.resizeRowsToContents()

    def _clear_plots_only(self):
        self.plot_title.clear()
        self.plot_title.hide()
        self.pred_plot_label.clear()
        self.pred_plot_label.hide()
        self.fft_title.hide()
        self.fft_plot_label.clear()
        self.fft_plot_label.hide()

    def _clear_details(self):
        self.metadata_text.clear()
        self._clear_plots_only()
        self.plot_placeholder.setText("Select a CSV file or evaluate checkpoints to view plots.")
        self.plot_placeholder.show()

    def _on_reset(self):
        self.viewer.clear()
        self.results = []
        self.summary_table.setRowCount(0)
        self._clear_details()
        self.statusBar().showMessage("Reset")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PatcherBotGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
