import sys
import os
import shutil
import random
import json
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QScrollArea, QGridLayout, 
                             QGroupBox, QMessageBox, QRadioButton, QDialog,
                             QTextEdit, QDialogButtonBox, QProgressBar,
                             QDoubleSpinBox, QSpinBox)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

# This should point to the updated cluster_faces.py file
from cluster_faces import (generate_and_cache_embeddings, cluster_embeddings, 
                           get_image_paths)

# --- GUI Configuration ---
SAMPLES_PER_CLUSTER = 5
THUMBNAIL_SIZE = 100
INSPECTOR_THUMBNAIL_SIZE = 128

class LazyImageLabel(QLabel):
    """A QLabel that loads its pixmap lazily to improve performance."""
    def __init__(self, file_path, size):
        super().__init__()
        self.file_path = file_path
        self.target_size = size
        self.is_selected = True
        self.pixmap_loaded = False
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #333; border: 1px solid #555;")

    def load_pixmap(self):
        if not self.pixmap_loaded and os.path.exists(self.file_path):
            try:
                pixmap = QPixmap(self.file_path)
                scaled_pixmap = pixmap.scaled(self.target_size, self.target_size, 
                                              Qt.AspectRatioMode.KeepAspectRatio, 
                                              Qt.TransformationMode.SmoothTransformation)
                self.setPixmap(scaled_pixmap)
                self.pixmap_loaded = True
                self.update_style()
            except Exception as e:
                self.setText("Load\nError")
                print(f"Error loading pixmap for {self.file_path}: {e}")
        elif not os.path.exists(self.file_path):
            self.setText("File\nNot Found")

    def mousePressEvent(self, event):
        self.is_selected = not self.is_selected
        self.update_style()
        super().mousePressEvent(event)

    def update_style(self):
        if not self.pixmap_loaded: return
        border_color = "#2ecc71" if self.is_selected else "#e74c3c"
        self.setStyleSheet(f"background-color: transparent; border: 2px solid {border_color};")

class ClusterInspectorWindow(QDialog):
    """Dialog to view/edit a cluster, with lazy loading for performance."""
    selection_confirmed = pyqtSignal(list)

    def __init__(self, cluster_title, file_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Inspector: {cluster_title}")
        self.setGeometry(150, 150, 1200, 800)
        self.setModal(True)
        self.image_widgets = []

        main_layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.grid_layout = QGridLayout(scroll_content)
        self.scroll_area.setWidget(scroll_content)
        main_layout.addWidget(self.scroll_area)

        row, col, MAX_COLS = 0, 0, 8
        for path in file_paths:
            img_widget = LazyImageLabel(path, INSPECTOR_THUMBNAIL_SIZE)
            self.image_widgets.append(img_widget)
            self.grid_layout.addWidget(img_widget, row, col)
            col = (col + 1) % MAX_COLS
            if col == 0: row += 1
        
        bottom_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("Confirm Selection")
        self.confirm_btn.clicked.connect(self.confirm_and_close)
        bottom_layout.addStretch(); bottom_layout.addWidget(self.confirm_btn)
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

        QTimer.singleShot(50, self.update_visible_images)
        self.scroll_area.verticalScrollBar().valueChanged.connect(self.update_visible_images)

    def update_visible_images(self, _=None):
        visible_top = self.scroll_area.verticalScrollBar().value()
        visible_bottom = visible_top + self.scroll_area.viewport().height()
        for widget in self.image_widgets:
            if not widget.pixmap_loaded:
                widget_top = widget.y()
                widget_bottom = widget_top + widget.height()
                if widget_top < (visible_bottom + 200) and widget_bottom > (visible_top - 200):
                    widget.load_pixmap()

    def confirm_and_close(self):
        approved_files = [widget.file_path for widget in self.image_widgets if widget.is_selected]
        self.selection_confirmed.emit(approved_files)
        self.accept()

class WorkerThread(QThread):
    status = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object, object)

    def __init__(self, image_folder, use_mtcnn, epsilon, min_samples):
        super().__init__()
        self.image_folder = image_folder
        self.use_mtcnn = use_mtcnn
        self.epsilon = epsilon
        self.min_samples = min_samples

    def run(self):
        try:
            self.status.emit("Step 1/3: Finding image files...")
            image_paths = get_image_paths(self.image_folder)
            if not image_paths: self.finished.emit(None, None); return
            
            self.status.emit(f"Step 2/3: Generating embeddings for {len(image_paths)} images...")
            embeddings_dict, rejected_files = generate_and_cache_embeddings(
                image_paths, 
                progress_callback=self.progress.emit, 
                use_mtcnn=self.use_mtcnn
            )
            if not embeddings_dict: self.finished.emit(None, rejected_files); return

            self.status.emit("Step 3/3: Clustering faces...")
            clusters, _ = cluster_embeddings(embeddings_dict, self.epsilon, self.min_samples)
            
            self.status.emit("Clustering complete. Displaying results.")
            self.finished.emit(clusters, rejected_files)
        except Exception as e:
            self.status.emit(f"An error occurred: {e}"); import traceback; traceback.print_exc(); self.finished.emit(None, None)

class FaceSorterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.clusters = {}; self.cluster_data = {}; self.input_folder = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Face Identity Sorter'); self.setGeometry(100, 100, 1400, 900)
        main_layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        self.select_folder_btn = QPushButton("Select Image Folder"); self.select_folder_btn.clicked.connect(self.select_folder)
        self.start_btn = QPushButton("Start Clustering"); self.start_btn.clicked.connect(self.start_clustering); self.start_btn.setEnabled(False)
        self.folder_label = QLabel("No folder selected.")
        self.load_project_btn = QPushButton("Load Project"); self.load_project_btn.clicked.connect(self.load_project)
        self.save_project_btn = QPushButton("Save Project"); self.save_project_btn.clicked.connect(self.save_project); self.save_project_btn.setEnabled(False)
        
        top_layout.addWidget(self.select_folder_btn)
        top_layout.addWidget(self.start_btn)
        top_layout.addWidget(self.folder_label, 1)
        top_layout.addWidget(self.load_project_btn)
        top_layout.addWidget(self.save_project_btn)
        main_layout.addLayout(top_layout)

        settings_area_layout = QHBoxLayout()
        processing_mode_group = QGroupBox("Processing Mode")
        processing_layout = QVBoxLayout()
        self.radio_pre_aligned = QRadioButton("Process Pre-aligned Face Crops (Recommended, Fastest)")
        self.radio_detect_faces = QRadioButton("Detect Faces in Full Images (Slower)")
        self.radio_pre_aligned.setChecked(True)
        processing_layout.addWidget(self.radio_pre_aligned)
        processing_layout.addWidget(self.radio_detect_faces)
        processing_mode_group.setLayout(processing_layout)
        settings_area_layout.addWidget(processing_mode_group, 1)

        adv_settings_group = QGroupBox("Advanced Clustering Settings")
        adv_settings_group.setCheckable(True)
        adv_settings_group.setChecked(False)
        adv_settings_layout = QGridLayout()
        adv_settings_layout.addWidget(QLabel("DBSCAN Epsilon:"), 0, 0)
        self.epsilon_spinbox = QDoubleSpinBox(); self.epsilon_spinbox.setRange(0.01, 1.0); self.epsilon_spinbox.setSingleStep(0.01); self.epsilon_spinbox.setValue(0.09)
        adv_settings_layout.addWidget(self.epsilon_spinbox, 0, 1)
        adv_settings_layout.addWidget(QLabel("DBSCAN Min Samples:"), 1, 0)
        self.min_samples_spinbox = QSpinBox(); self.min_samples_spinbox.setRange(2, 100); self.min_samples_spinbox.setValue(2)
        adv_settings_layout.addWidget(self.min_samples_spinbox, 1, 1)
        adv_settings_group.setLayout(adv_settings_layout)
        settings_area_layout.addWidget(adv_settings_group)
        main_layout.addLayout(settings_area_layout)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Welcome! Select a folder and click Start.")
        status_layout.addWidget(self.status_label, 1)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout)

        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget(); self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content); main_layout.addWidget(self.scroll_area, 1)
        
        # --- Bottom Controls ---
        bottom_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All"); self.select_all_btn.clicked.connect(lambda: self.toggle_all_clusters(True))
        self.deselect_all_btn = QPushButton("Deselect All"); self.deselect_all_btn.clicked.connect(lambda: self.toggle_all_clusters(False))
        
        output_group = QGroupBox("Output Format")
        output_group_layout = QVBoxLayout()
        self.radio_separate = QRadioButton("Save in separate folders (cluster_0, cluster_1, ...)")
        self.radio_single = QRadioButton("Save all selected images into one folder")
        self.radio_separate.setChecked(True)
        output_group_layout.addWidget(self.radio_separate)
        output_group_layout.addWidget(self.radio_single)
        output_group.setLayout(output_group_layout)

        self.save_clusters_btn = QPushButton("Save Selected Clusters"); self.save_clusters_btn.clicked.connect(self.save_selected_clusters); self.save_clusters_btn.setEnabled(False)
        
        bottom_layout.addWidget(self.select_all_btn)
        bottom_layout.addWidget(self.deselect_all_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(output_group)
        bottom_layout.addWidget(self.save_clusters_btn)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, 'resize_timer'):
            self.resize_timer = QTimer()
            self.resize_timer.setSingleShot(True)
            self.resize_timer.timeout.connect(self.reflow_clusters)
        self.resize_timer.start(100)

    def reflow_clusters(self):
        if not self.cluster_data: return
        
        sorted_data_items = sorted(self.cluster_data.items(), key=lambda item: len(item[1]['original_files']), reverse=True)
        widgets = [item[1]['widget'] for item in sorted_data_items]

        for widget in widgets:
            self.grid_layout.removeWidget(widget)
            widget.setParent(None)

        width = self.scroll_area.width()
        num_cols = max(1, 3 if width > 1300 else 2 if width > 850 else 1)

        row, col = 0, 0
        for widget in widgets:
            self.grid_layout.addWidget(widget, row, col)
            col = (col + 1) % num_cols
            if col == 0: row += 1

    def display_clusters(self):
        self.clear_grid()
        if not self.clusters: return

        for label, file_paths in self.clusters.items():
            title = f"Unclassified ({len(file_paths)})" if label == -1 else f"Cluster {label} ({len(file_paths)})"
            group_box = ClickableGroupBox(title)
            image_layout = QHBoxLayout(group_box.image_container)
            sample_paths = random.sample(file_paths, min(len(file_paths), SAMPLES_PER_CLUSTER))
            for path in sample_paths:
                img_label = QLabel(); pixmap = QPixmap(path)
                img_label.setPixmap(pixmap.scaled(THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                image_layout.addWidget(img_label)
            
            self.cluster_data[label] = {'widget': group_box, 'original_files': file_paths, 'approved_files': list(file_paths)}
            group_box.view_requested.connect(lambda checked=False, l=label: self.open_inspector(l))
            
        self.reflow_clusters()

    def start_clustering(self):
        if not self.input_folder: QMessageBox.warning(self, "Warning", "Please select a folder first."); return
        
        self.set_controls_enabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.clear_grid()

        use_mtcnn = self.radio_detect_faces.isChecked()
        epsilon = self.epsilon_spinbox.value()
        min_samples = self.min_samples_spinbox.value()

        self.worker = WorkerThread(self.input_folder, use_mtcnn, epsilon, min_samples)
        self.worker.status.connect(self.status_label.setText)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_clustering_finished)
        self.worker.start()

    def on_clustering_finished(self, clusters, rejected_files):
        self.set_controls_enabled(True)
        self.progress_bar.setVisible(False)
        
        if clusters is None: 
            self.status_label.setText("Clustering failed. Check console for errors.")
        else: 
            self.clusters = clusters
            self.display_clusters()

    def save_project(self):
        if not self.cluster_data: QMessageBox.warning(self, "Nothing to Save", "Please run clustering before saving."); return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project File", "", "JSON Files (*.json)")
        if not file_path: return
        
        project_data = {'input_folder': self.input_folder, 'clusters': {}}
        for label, data in self.cluster_data.items():
            project_data['clusters'][str(label)] = {'is_selected': data['widget'].isChecked(), 'approved_files': data['approved_files']}
        try:
            with open(file_path, 'w') as f: json.dump(project_data, f, indent=4)
            QMessageBox.information(self, "Success", f"Project saved to:\n{file_path}")
        except Exception as e: QMessageBox.critical(self, "Error", f"Could not save project file.\nError: {e}")

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Project File", "", "JSON Files (*.json)")
        if not file_path: return
        try:
            with open(file_path, 'r') as f: project_data = json.load(f)
        except Exception as e: QMessageBox.critical(self, "Error", f"Could not read project file.\nError: {e}"); return
        
        if not self.cluster_data or os.path.normpath(project_data.get('input_folder')) != os.path.normpath(self.input_folder):
            QMessageBox.warning(self, "Folder Mismatch", "To load a project, first select the correct image folder and run the initial clustering. Then, you can load the project to restore your selections."); return

        for label_str, data in project_data.get('clusters', {}).items():
            label = int(label_str)
            if label in self.cluster_data:
                self.cluster_data[label]['widget'].setChecked(data.get('is_selected', False))
                self.cluster_data[label]['approved_files'] = data.get('approved_files', [])
                self.update_cluster_title(label)
        QMessageBox.information(self, "Success", "Project selections and approvals have been loaded.")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder: self.input_folder = folder; self.folder_label.setText(f"Folder: ...{os.path.basename(folder)}"); self.start_btn.setEnabled(True)

    def open_inspector(self, cluster_label):
        data = self.cluster_data[cluster_label]
        inspector = ClusterInspectorWindow(data['widget'].title(), data['original_files'], self)
        inspector.selection_confirmed.connect(lambda files: self.update_approved_files(cluster_label, files))
        inspector.exec()

    def update_approved_files(self, cluster_label, approved_files):
        self.cluster_data[cluster_label]['approved_files'] = approved_files
        self.update_cluster_title(cluster_label)

    def update_cluster_title(self, cluster_label):
        data = self.cluster_data[cluster_label]
        widget, original_count, approved_count = data['widget'], len(data['original_files']), len(data['approved_files'])
        base_title = "Unclassified" if cluster_label == -1 else f"Cluster {cluster_label}"
        title = f"{base_title} ({approved_count}/{original_count} selected)" if approved_count != original_count else f"{base_title} ({original_count} faces)"
        widget.setTitle(title)

    def save_selected_clusters(self):
        output_folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save To")
        if not output_folder: return
        
        images_copied = 0
        selected_count = 0

        if self.radio_separate.isChecked():
            for label, data in self.cluster_data.items():
                if data['widget'].isChecked():
                    selected_count += 1
                    folder_name = "unclassified" if label == -1 else f"cluster_{label}"
                    cluster_dir = os.path.join(output_folder, folder_name)
                    os.makedirs(cluster_dir, exist_ok=True)
                    for file_path in data['approved_files']:
                        try: shutil.copy(file_path, cluster_dir); images_copied += 1
                        except Exception as e: print(f"Could not copy {file_path}: {e}")
            QMessageBox.information(self, "Success", f"Saved {selected_count} clusters ({images_copied} images) to separate folders.")
        else: # Save to single folder
            for label, data in self.cluster_data.items():
                if data['widget'].isChecked():
                    selected_count += 1
                    for file_path in data['approved_files']:
                        try: shutil.copy(file_path, output_folder); images_copied += 1
                        except Exception as e: print(f"Could not copy {file_path}: {e}")
            QMessageBox.information(self, "Success", f"Saved {selected_count} clusters ({images_copied} images) into the single folder.")

    def set_controls_enabled(self, enabled):
        self.start_btn.setEnabled(enabled)
        self.select_folder_btn.setEnabled(enabled)
        self.load_project_btn.setEnabled(enabled)
        self.save_project_btn.setEnabled(enabled)
        self.save_clusters_btn.setEnabled(enabled)

    def update_progress(self, current, total):
        if total > 0: self.progress_bar.setValue(int((current / total) * 100))

    def toggle_all_clusters(self, state):
        for data in self.cluster_data.values(): data['widget'].setChecked(state)

    def clear_grid(self):
        if hasattr(self, 'cluster_data'):
            for data in self.cluster_data.values():
                data['widget'].deleteLater()
        self.cluster_data = {}

class ClickableGroupBox(QGroupBox):
    clicked = pyqtSignal(); view_requested = pyqtSignal()
    def __init__(self, title=""):
        super().__init__(title)
        self.setCheckable(True); self.setChecked(False)
        self.setStyleSheet("""
            ClickableGroupBox { border: 1px solid gray; border-radius: 5px; margin-top: 1ex; }
            ClickableGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }
            ClickableGroupBox[checked="true"] { border: 2px solid #3498db; font-weight: bold; }
        """)
        self.box_layout = QVBoxLayout()
        self.image_container = QWidget()
        self.box_layout.addWidget(self.image_container)
        self.view_button = QPushButton("View / Edit Selection")
        self.view_button.clicked.connect(self.view_requested.emit)
        self.box_layout.addWidget(self.view_button); self.setLayout(self.box_layout)
    def mousePressEvent(self, event):
        self.setChecked(not self.isChecked()); self.clicked.emit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceSorterApp()
    ex.show()
    sys.exit(app.exec())
