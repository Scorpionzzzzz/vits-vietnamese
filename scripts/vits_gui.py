#!/usr/bin/env python3
"""
Professional VITS GUI Controller
Modern interface for VITS model parameter tuning
"""

import sys
import os
import torch
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QSlider, QSpinBox, QDoubleSpinBox, 
    QPushButton, QTextEdit, QFileDialog, QProgressBar, QGroupBox,
    QCheckBox, QComboBox, QLineEdit, QTabWidget, QSplitter,
    QMessageBox, QStatusBar, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
from transformers import VitsModel, AutoTokenizer
import qdarktheme

class VitsWorker(QThread):
    """Worker thread for VITS model operations"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, model_dir, text, params, output_dir):
        super().__init__()
        self.model_dir = model_dir
        self.text = text
        self.params = params
        self.output_dir = output_dir
        
    def run(self):
        try:
            self.progress.emit("🔄 Đang load model...")
            
            # Load model
            model = VitsModel.from_pretrained(self.model_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
            # Set parameters
            model.config.noise_scale = self.params['noise_scale']
            model.config.noise_scale_duration = self.params['noise_scale_duration']
            model.config.speaking_rate = self.params['speaking_rate']
            
            # Set seed
            seed = self.params['seed']
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            self.progress.emit("🎵 Đang tạo audio...")
            
            # Add padding
            padded_text = self.params['padding_start'] + self.text + self.params['padding_end']
            inputs = tokenizer(padded_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model(**inputs)
            
            waveform = output.waveform.squeeze().cpu().numpy()
            sampling_rate = model.config.sampling_rate
            
            # Save audio
            output_file = Path(self.output_dir) / f"vits_output_{seed}.wav"
            wav.write(str(output_file), sampling_rate, waveform)
            
            duration = len(waveform) / sampling_rate
            self.finished.emit(f"✅ Hoàn thành! Audio: {output_file.name} ({duration:.2f}s)")
            
        except Exception as e:
            self.error.emit(f"❌ Lỗi: {str(e)}")

class ModernSlider(QWidget):
    """Custom modern slider widget"""
    def __init__(self, label, min_val, max_val, default, decimals=2, unit=""):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Label
        label_widget = QLabel(label)
        label_widget.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
                margin-bottom: 5px;
            }
        """)
        layout.addWidget(label_widget)
        
        # Slider and value display
        slider_layout = QHBoxLayout()
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * 100))
        self.slider.setMaximum(int(max_val * 100))
        self.slider.setValue(int(default * 100))
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 2px solid #2980b9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #3498db;
                border-radius: 4px;
            }
        """)
        
        if decimals == 0:
            self.value_display = QSpinBox()
            self.value_display.setMinimum(int(min_val))
            self.value_display.setMaximum(int(max_val))
            self.value_display.setValue(int(default))
        else:
            self.value_display = QDoubleSpinBox()
            self.value_display.setMinimum(min_val)
            self.value_display.setMaximum(max_val)
            self.value_display.setValue(default)
            self.value_display.setDecimals(decimals)
        
        self.value_display.setSuffix(unit)
        self.value_display.setStyleSheet("""
            QSpinBox, QDoubleSpinBox {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                padding: 5px;
                background: white;
                font-weight: bold;
                color: #2c3e50;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #3498db;
            }
        """)
        
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.value_display)
        layout.addLayout(slider_layout)
        
        # Connect signals
        self.slider.valueChanged.connect(self._slider_changed)
        self.value_display.valueChanged.connect(self._display_changed)
        
    def _slider_changed(self, value):
        if isinstance(self.value_display, QDoubleSpinBox):
            self.value_display.setValue(value / 100)
        else:
            self.value_display.setValue(value)
            
    def _display_changed(self, value):
        if isinstance(self.value_display, QDoubleSpinBox):
            self.slider.setValue(int(value * 100))
        else:
            self.slider.setValue(value)
    
    def get_value(self):
        return self.value_display.value()

class VitsGUI(QMainWindow):
    """Main VITS GUI application"""
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🎵 VITS Professional Controller")
        self.setGeometry(100, 100, 1200, 800)
        
        # Apply dark theme
        qdarktheme.setup_theme("dark")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Model and Parameters
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Text input and Output
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("🚀 Sẵn sàng - Vui lòng chọn model")
        
        # Style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #34495e;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ecf0f1;
            }
            QPushButton {
                background-color: #3498db;
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f4e79;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
            QTextEdit {
                border: 2px solid #34495e;
                border-radius: 5px;
                background-color: #34495e;
                color: #ecf0f1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QLineEdit {
                border: 2px solid #34495e;
                border-radius: 5px;
                padding: 8px;
                background-color: #34495e;
                color: #ecf0f1;
                font-size: 12px;
            }
        """)
        
    def create_left_panel(self):
        """Create left panel with model and parameters"""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        
        # Model Selection
        model_group = QGroupBox("🔧 Model Configuration")
        model_layout = QVBoxLayout(model_group)
        
        # Model path
        path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Chọn đường dẫn đến model VITS...")
        self.model_path_edit.setText(r"D:\Workspace\NLP\TEXT-TO-SPEECH\finetune-hf-vits\runs\mms_vie_ft_single_spk")
        
        browse_btn = QPushButton("📁 Browse")
        browse_btn.clicked.connect(self.browse_model)
        
        path_layout.addWidget(self.model_path_edit)
        path_layout.addWidget(browse_btn)
        model_layout.addLayout(path_layout)
        
        # Load model button
        self.load_btn = QPushButton("🚀 Load Model")
        self.load_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_btn)
        
        layout.addWidget(model_group)
        
        # Model Info
        self.model_info_group = QGroupBox("📊 Model Information")
        self.model_info_layout = QVBoxLayout(self.model_info_group)
        self.model_info_layout.addWidget(QLabel("Chưa load model"))
        layout.addWidget(self.model_info_group)
        
        # Parameters
        params_group = QGroupBox("🎛️ Generation Parameters")
        params_layout = QGridLayout(params_group)
        
        # Noise scale
        self.noise_scale_slider = ModernSlider("Noise Scale", 0.1, 2.0, 0.667, 3)
        params_layout.addWidget(self.noise_scale_slider, 0, 0)
        
        # Noise scale duration
        self.noise_duration_slider = ModernSlider("Noise Duration", 0.1, 2.0, 0.8, 3)
        params_layout.addWidget(self.noise_duration_slider, 1, 0)
        
        # Speaking rate
        self.speaking_rate_slider = ModernSlider("Speaking Rate", 0.5, 2.0, 1.0, 2)
        params_layout.addWidget(self.speaking_rate_slider, 2, 0)
        
        # Seed
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setMinimum(1)
        self.seed_spinbox.setMaximum(999999)
        self.seed_spinbox.setValue(42)
        self.seed_spinbox.setStyleSheet("""
            QSpinBox {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                padding: 8px;
                background: white;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed:"))
        seed_layout.addWidget(self.seed_spinbox)
        params_layout.addLayout(seed_layout, 3, 0)
        
        layout.addWidget(params_group)
        
        # Padding Configuration
        padding_group = QGroupBox("📝 Text Padding")
        padding_layout = QGridLayout(padding_group)
        
        # Padding start
        self.padding_start_edit = QLineEdit("  ")
        self.padding_start_edit.setPlaceholderText("Padding đầu")
        padding_layout.addWidget(QLabel("Padding Start:"), 0, 0)
        padding_layout.addWidget(self.padding_start_edit, 0, 1)
        
        # Padding end
        self.padding_end_edit = QLineEdit(" . ")
        self.padding_end_edit.setPlaceholderText("Padding cuối")
        padding_layout.addWidget(QLabel("Padding End:"), 1, 0)
        padding_layout.addWidget(self.padding_end_edit, 1, 1)
        
        layout.addWidget(padding_group)
        
        # Generate button
        self.generate_btn = QPushButton("🎵 Generate Audio")
        self.generate_btn.clicked.connect(self.generate_audio)
        self.generate_btn.setEnabled(False)
        layout.addWidget(self.generate_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        return left_widget
        
    def create_right_panel(self):
        """Create right panel with text input and output"""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        # Text Input
        text_group = QGroupBox("📝 Text Input")
        text_layout = QVBoxLayout(text_group)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Nhập text tiếng Việt để chuyển thành speech...")
        self.text_edit.setMaximumHeight(150)
        text_layout.addWidget(self.text_edit)
        
        # Quick text buttons
        quick_text_layout = QHBoxLayout()
        quick_texts = [
            "Xin chào, đây là mô hình TTS tiếng Việt.",
            "Hôm nay trời đẹp quá, mình muốn đi dạo.",
            "Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh.",
            "Chúc các bạn ngủ ngon nhá!"
        ]
        
        for text in quick_texts:
            btn = QPushButton(text[:20] + "...")
            btn.setMaximumWidth(120)
            btn.clicked.connect(lambda checked, t=text: self.text_edit.setText(t))
            quick_text_layout.addWidget(btn)
            
        text_layout.addLayout(quick_text_layout)
        layout.addWidget(text_group)
        
        # Output Configuration
        output_group = QGroupBox("💾 Output Settings")
        output_layout = QVBoxLayout(output_group)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(r"D:\Workspace\NLP\TEXT-TO-SPEECH\generated_audio")
        self.output_dir_edit.setPlaceholderText("Thư mục lưu audio...")
        
        output_browse_btn = QPushButton("📁 Browse")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(output_browse_btn)
        output_layout.addLayout(output_dir_layout)
        
        layout.addWidget(output_group)
        
        # Log Output
        log_group = QGroupBox("📋 Generation Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        # Clear log button
        clear_log_btn = QPushButton("🧹 Clear Log")
        clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_btn)
        
        layout.addWidget(log_group)
        
        return right_widget
        
    def browse_model(self):
        """Browse for model directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục model VITS")
        if dir_path:
            self.model_path_edit.setText(dir_path)
            
    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu audio")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            
    def load_model(self):
        """Load VITS model"""
        model_dir = self.model_path_edit.text()
        if not os.path.exists(model_dir):
            QMessageBox.critical(self, "Lỗi", "Thư mục model không tồn tại!")
            return
            
        try:
            self.log_message("🔄 Đang load model...")
            self.load_btn.setEnabled(False)
            self.load_btn.setText("⏳ Loading...")
            
            # Load model in thread
            self.model = VitsModel.from_pretrained(model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Update UI
            self.load_btn.setText("✅ Model Loaded")
            self.generate_btn.setEnabled(True)
            self.status_bar.showMessage("🎵 Model đã sẵn sàng")
            
            # Update model info
            self.update_model_info()
            
            self.log_message("✅ Model đã load thành công!")
            
        except Exception as e:
            self.log_message(f"❌ Lỗi load model: {str(e)}")
            self.load_btn.setEnabled(True)
            self.load_btn.setText("🚀 Load Model")
            
    def update_model_info(self):
        """Update model information display"""
        if self.model:
            # Clear old info
            for i in reversed(range(self.model_info_layout.count())):
                self.model_info_layout.itemAt(i).widget().setParent(None)
                
            # Add new info
            info_texts = [
                f"📊 Model Type: {self.model.config.model_type}",
                f"🔧 Vocab Size: {self.model.config.vocab_size}",
                f"🎵 Sampling Rate: {self.model.config.sampling_rate} Hz",
                f"🎭 Speakers: {self.model.config.num_speakers}",
                f"📏 Hidden Size: {self.model.config.hidden_size}",
                f"🔍 Layers: {self.model.config.num_hidden_layers}"
            ]
            
            for text in info_texts:
                label = QLabel(text)
                label.setStyleSheet("color: #ecf0f1; font-size: 11px;")
                self.model_info_layout.addWidget(label)
                
    def generate_audio(self):
        """Generate audio from text"""
        if not self.model or not self.tokenizer:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng load model trước!")
            return
            
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập text!")
            return
            
        # Get parameters
        params = {
            'noise_scale': self.noise_scale_slider.get_value(),
            'noise_scale_duration': self.noise_duration_slider.get_value(),
            'speaking_rate': self.speaking_rate_slider.get_value(),
            'seed': self.seed_spinbox.value(),
            'padding_start': self.padding_start_edit.text(),
            'padding_end': self.padding_end_edit.text()
        }
        
        # Start worker thread
        self.worker = VitsWorker(
            self.model_path_edit.text(),
            text,
            params,
            self.output_dir_edit.text()
        )
        
        self.worker.progress.connect(self.log_message)
        self.worker.finished.connect(self.generation_finished)
        self.worker.error.connect(self.generation_error)
        
        # Update UI
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("⏳ Generating...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        self.worker.start()
        
    def generation_finished(self, message):
        """Handle generation completion"""
        self.log_message(message)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("🎵 Generate Audio")
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("✅ Audio đã được tạo thành công!")
        
    def generation_error(self, error_message):
        """Handle generation error"""
        self.log_message(error_message)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("🎵 Generate Audio")
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("❌ Có lỗi xảy ra!")
        
    def log_message(self, message):
        """Add message to log"""
        self.log_text.append(f"[{QTimer().remainingTime()}] {message}")
        self.log_text.ensureCursorVisible()
        
    def clear_log(self):
        """Clear log text"""
        self.log_text.clear()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("VITS Professional Controller")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("VITS TTS")
    
    # Create and show main window
    window = VitsGUI()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
