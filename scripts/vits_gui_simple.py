#!/usr/bin/env python3
"""
Simple VITS GUI Controller
Clean interface for VITS model parameter tuning without external themes
"""

import sys
import os
import torch
import numpy as np
import scipy.io.wavfile as wav
import json
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QSlider, QSpinBox, QDoubleSpinBox, 
    QPushButton, QTextEdit, QFileDialog, QProgressBar, QGroupBox,
    QLineEdit, QMessageBox, QStatusBar, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QFont
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl
from transformers import VitsModel, AutoTokenizer

class AudioPlayer:
    """Audio player for playing generated audio files"""
    def __init__(self):
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.current_file = None
        
    def play(self, file_path):
        """Play audio file"""
        if os.path.exists(file_path):
            self.current_file = file_path
            self.player.setSource(QUrl.fromLocalFile(file_path))
            self.player.play()
            
    def stop(self):
        """Stop audio playback"""
        self.player.stop()
        
    def is_playing(self):
        """Check if audio is currently playing"""
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

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
            

            
            self.progress.emit("🎵 Đang tạo audio...")
            
            # Add padding
            padded_text = self.params['padding_start'] + self.text + self.params['padding_end']
            inputs = tokenizer(padded_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model(**inputs)
            
            waveform = output.waveform.squeeze().cpu().numpy()
            sampling_rate = model.config.sampling_rate
            
            # Save audio with text-based filename (unique per text)
            # Create a simple filename from text content
            text_hash = str(hash(self.text))[-8:]  # Use last 8 chars of hash
            safe_text = "".join(c for c in self.text[:20] if c.isalnum() or c.isspace()).strip()
            safe_text = safe_text.replace(" ", "_")[:15]  # Limit length and replace spaces
            
            output_file = Path(self.output_dir) / f"vits_{safe_text}_{text_hash}.wav"
            wav.write(str(output_file), sampling_rate, waveform)
            
            duration = len(waveform) / sampling_rate
            result_info = {
                'file_path': str(output_file),
                'duration': duration,
                'text': self.text,
                'params': self.params
            }
            self.finished.emit(json.dumps(result_info))
            
        except Exception as e:
            self.error.emit(f"❌ Lỗi: {str(e)}")

class AudioHistory:
    """Manage audio generation history"""
    def __init__(self, history_file="audio_history.json"):
        self.history_file = history_file
        self.history = self.load_history()
        
    def load_history(self):
        """Load history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except:
            return []
            
    def save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
            
    def add_entry(self, text, file_path, params):
        """Add new audio entry to history - one per text"""
        # Check if text already exists
        for entry in self.history:
            if entry['text'] == text:
                # Update existing entry (same text = same audio)
                entry['file_path'] = file_path
                entry['params'] = params
                entry['timestamp'] = datetime.now().isoformat()
                self.save_history()
                return False  # Not new, just updated
        
        # Add new entry only if text doesn't exist
        new_entry = {
            'text': text,
            'file_path': file_path,
            'params': params,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(new_entry)
        self.save_history()
        return True  # New entry
        
    def get_history(self):
        """Get all history entries"""
        return self.history
        
    def clear_history(self):
        """Clear all history"""
        self.history = []
        self.save_history()

class ModernSlider(QWidget):
    """Custom modern slider widget"""
    def __init__(self, label, min_val, max_val, default, decimals=2, unit=""):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Label
        label_widget = QLabel(label)
        label_widget.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(label_widget)
        
        # Slider and value display
        slider_layout = QHBoxLayout()
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * 100))
        self.slider.setMaximum(int(max_val * 100))
        self.slider.setValue(int(default * 100))
        
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
        
    def set_value(self, value):
        """Set slider value"""
        self.value_display.setValue(value)
        if isinstance(self.value_display, QDoubleSpinBox):
            self.slider.setValue(int(value * 100))
        else:
            self.slider.setValue(value)

class VitsGUI(QMainWindow):
    """Main VITS GUI application"""
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.worker = None
        self.audio_player = AudioPlayer()
        self.audio_history = AudioHistory()
        self.current_audio_file = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🎵 VITS Professional Controller")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #333333;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QTextEdit {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: white;
                color: #333333;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QLineEdit {
                border: 2px solid #cccccc;
                border-radius: 5px;
                padding: 8px;
                background-color: white;
                color: #333333;
                font-size: 12px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #cccccc;
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 2px solid #45a049;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border-radius: 4px;
            }
        """)
        
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
        
        # Load initial history
        self.refresh_history()
        
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
        self.model_path_edit.setText(r"/home/huathanh/WorkSpaces/NLP/TEXT-TO-SPEECH/finetune-hf-vits/runs/mms_vie_ft_single_spk")
        
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
        self.text_edit.textChanged.connect(self.on_text_changed)
        text_layout.addWidget(self.text_edit)
        
        # Text controls
        text_controls_layout = QHBoxLayout()
        
        # Clean text button
        clean_btn = QPushButton("🧹 Clean Text")
        clean_btn.clicked.connect(self.manual_clean_text)
        clean_btn.setToolTip("Làm sạch ký tự đặc biệt và định dạng lỗi")
        text_controls_layout.addWidget(clean_btn)
        
        text_controls_layout.addStretch()
        text_layout.addLayout(text_controls_layout)
        
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
        self.output_dir_edit.setText(r"/home/huathanh/WorkSpaces/NLP/TEXT-TO-SPEECH/generated_audio")
        self.output_dir_edit.setPlaceholderText("Thư mục lưu audio...")
        
        output_browse_btn = QPushButton("📁 Browse")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(output_browse_btn)
        output_layout.addLayout(output_dir_layout)
        
        layout.addWidget(output_group)
        
        # Audio Player Controls
        audio_group = QGroupBox("🎵 Audio Player")
        audio_layout = QVBoxLayout(audio_group)
        
        # Audio controls
        audio_controls = QHBoxLayout()
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        
        audio_controls.addWidget(self.play_btn)
        audio_controls.addWidget(self.stop_btn)
        audio_layout.addLayout(audio_controls)
        
        # Current audio info
        self.audio_info_label = QLabel("Chưa có audio")
        self.audio_info_label.setStyleSheet("color: #666; font-style: italic;")
        audio_layout.addWidget(self.audio_info_label)
        
        layout.addWidget(audio_group)
        
        # Audio History
        history_group = QGroupBox("📚 Text History (1 audio per text)")
        history_layout = QVBoxLayout(history_group)
        
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        self.history_list.itemDoubleClicked.connect(self.load_history_item)
        history_layout.addWidget(self.history_list)
        
        # History controls
        history_controls = QHBoxLayout()
        refresh_history_btn = QPushButton("🔄 Refresh")
        refresh_history_btn.clicked.connect(self.refresh_history)
        
        clear_history_btn = QPushButton("🗑️ Clear")
        clear_history_btn.clicked.connect(self.clear_history)
        
        history_controls.addWidget(refresh_history_btn)
        history_controls.addWidget(clear_history_btn)
        history_layout.addLayout(history_controls)
        
        layout.addWidget(history_group)
        
        # Log Output
        log_group = QGroupBox("📋 Generation Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
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
                label.setFont(QFont("Arial", 9))
                label.setStyleSheet("color: #333333; padding: 2px;")
                label.setWordWrap(True)
                self.model_info_layout.addWidget(label)
                
    def generate_audio(self):
        """Generate audio from text"""
        if not self.model or not self.tokenizer:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng load model trước!")
            return
            
        # Get and clean text
        raw_text = self.text_edit.toPlainText()
        text = self.clean_text(raw_text)
        
        if not text:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập text!")
            return
            
        # Update text field if cleaned
        if raw_text != text:
            self.text_edit.setPlainText(text)
            self.log_message("🧹 Text đã được làm sạch trước khi generate")
            
        # Get parameters
        params = {
            'noise_scale': self.noise_scale_slider.get_value(),
            'noise_scale_duration': self.noise_duration_slider.get_value(),
            'speaking_rate': self.speaking_rate_slider.get_value(),
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
        
    def generation_finished(self, result_json):
        """Handle generation completion"""
        try:
            result = json.loads(result_json)
            
            # Update current audio file
            self.current_audio_file = result['file_path']
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            # Update audio info
            filename = os.path.basename(result['file_path'])
            duration = result['duration']
            self.audio_info_label.setText(f"📁 {filename} ({duration:.2f}s)")
            
            # Add to history
            params = {
                'noise_scale': self.noise_scale_slider.get_value(),
                'noise_scale_duration': self.noise_duration_slider.get_value(),
                'speaking_rate': self.speaking_rate_slider.get_value(),
                'padding_start': self.padding_start_edit.text(),
                'padding_end': self.padding_end_edit.text()
            }
            
            is_new = self.audio_history.add_entry(result['text'], result['file_path'], params)
            if is_new:
                self.log_message(f"✅ Hoàn thành! Audio mới: {filename} ({duration:.2f}s)")
                self.refresh_history()  # Refresh history list
            else:
                self.log_message(f"✅ Hoàn thành! Audio cập nhật cho text đã có: {filename} ({duration:.2f}s)")
            
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("🎵 Generate Audio")
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("✅ Audio đã được tạo thành công!")
            
        except Exception as e:
            self.log_message(f"❌ Lỗi xử lý kết quả: {str(e)}")
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("🎵 Generate Audio")
            self.progress_bar.setVisible(False)
        
    def generation_error(self, error_message):
        """Handle generation error"""
        self.log_message(error_message)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("🎵 Generate Audio")
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("❌ Có lỗi xảy ra!")
        
    def log_message(self, message):
        """Add message to log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.ensureCursorVisible()
        
    def clear_log(self):
        """Clear log text"""
        self.log_text.clear()
        
    def on_text_changed(self):
        """Handle text change and clean special characters"""
        try:
            cursor = self.text_edit.textCursor()
            position = cursor.position()
            
            # Get current text
            text = self.text_edit.toPlainText()
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Only update if text changed
            if text != cleaned_text:
                self.text_edit.blockSignals(True)  # Prevent recursive calls
                self.text_edit.setPlainText(cleaned_text)
                
                # Restore cursor position
                cursor.setPosition(min(position, len(cleaned_text)))
                self.text_edit.setTextCursor(cursor)
                self.text_edit.blockSignals(False)
                
                self.log_message("🧹 Đã làm sạch text input")
        except Exception as e:
            print(f"Error in text cleaning: {e}")
            
    def clean_text(self, text):
        """Clean text input from special characters and formatting"""
        if not text:
            return text
            
        # Remove zero-width characters and unusual spaces
        import unicodedata
        
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove zero-width characters
        zero_width_chars = [
            '\u200b',  # zero-width space
            '\u200c',  # zero-width non-joiner
            '\u200d',  # zero-width joiner
            '\ufeff',  # byte order mark
            '\u2060',  # word joiner
        ]
        
        for char in zero_width_chars:
            text = text.replace(char, '')
            
        # Replace unusual spaces with normal space
        unusual_spaces = [
            '\u00a0',  # non-breaking space
            '\u2002',  # en space
            '\u2003',  # em space
            '\u2009',  # thin space
            '\u202f',  # narrow no-break space
        ]
        
        for space in unusual_spaces:
            text = text.replace(space, ' ')
            
        # Clean up multiple spaces
        while '  ' in text:
            text = text.replace('  ', ' ')
            
        # Remove leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        return text.strip()
        
    def manual_clean_text(self):
        """Manually clean text on button click"""
        try:
            raw_text = self.text_edit.toPlainText()
            if not raw_text.strip():
                self.log_message("⚠️ Không có text để làm sạch")
                return
                
            cleaned_text = self.clean_text(raw_text)
            
            if raw_text != cleaned_text:
                self.text_edit.blockSignals(True)
                self.text_edit.setPlainText(cleaned_text)
                self.text_edit.blockSignals(False)
                self.log_message("✅ Text đã được làm sạch thành công!")
            else:
                self.log_message("✅ Text đã sạch, không cần thay đổi")
        except Exception as e:
            self.log_message(f"❌ Lỗi khi làm sạch text: {str(e)}")
        
    def play_audio(self):
        """Play current audio file"""
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            self.audio_player.play(self.current_audio_file)
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.log_message("🎵 Đang phát audio...")
            
    def stop_audio(self):
        """Stop audio playback"""
        self.audio_player.stop()
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_message("⏹️ Đã dừng audio")
        
    def refresh_history(self):
        """Refresh audio history list"""
        self.history_list.clear()
        history = self.audio_history.get_history()
        
        for entry in history:
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
            text_preview = entry['text'][:50] + "..." if len(entry['text']) > 50 else entry['text']
            item_text = f"[{timestamp}] {text_preview}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.history_list.addItem(item)
            
        self.log_message(f"📚 Đã load {len(history)} text unique trong lịch sử")
        
    def clear_history(self):
        """Clear audio history"""
        reply = QMessageBox.question(
            self, "Xác nhận", "Bạn có chắc muốn xóa toàn bộ lịch sử?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.audio_history.clear_history()
            self.history_list.clear()
            self.log_message("🗑️ Đã xóa toàn bộ lịch sử")
            
    def load_history_item(self, item):
        """Load selected history item"""
        entry = item.data(Qt.ItemDataRole.UserRole)
        if entry:
            # Load text
            self.text_edit.setText(entry['text'])
            
            # Load parameters
            params = entry['params']
            # Update sliders properly
            self.noise_scale_slider.set_value(params['noise_scale'])
            self.noise_duration_slider.set_value(params['noise_scale_duration'])
            self.speaking_rate_slider.set_value(params['speaking_rate'])
            self.padding_start_edit.setText(params['padding_start'])
            self.padding_end_edit.setText(params['padding_end'])
            
            # Load audio file
            if os.path.exists(entry['file_path']):
                self.current_audio_file = entry['file_path']
                self.play_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                
                filename = os.path.basename(entry['file_path'])
                duration = entry.get('duration', 0)
                self.audio_info_label.setText(f"📁 {filename} ({duration:.2f}s)")
                
                self.log_message(f"📚 Đã load audio từ lịch sử: {filename}")
            else:
                self.log_message("⚠️ File audio không tồn tại trong lịch sử")

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
