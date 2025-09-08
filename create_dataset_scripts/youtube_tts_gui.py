#!/usr/bin/env python3
"""
YouTube to VITS GUI
Graphical interface for the YouTube to VITS dataset pipeline
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import subprocess
import queue
import time

# Import our pipeline modules
from download_youtube import YouTubeDownloader
from separate_voice import VoiceSeparator
from transcribe_dataset import TranscriptionProcessor
from incremental_update import separate_new, append_transcribe_and_split

class YouTubeTTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube → VITS Vietnamese Dataset Builder")
        self.root.geometry("980x720")
        # Resolve workspace base (project root = parent of this scripts dir)
        self.base_dir = Path(__file__).resolve().parents[1]
        
        # Setup logging queue
        self.log_queue = queue.Queue()
        self.setup_gui()
        self.setup_logging()
        
        # Initialize processors
        self.downloader = None
        self.separator = None
        self.transcriber = None
        
        # Check directories
        self.check_directories()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="YouTube → VITS Vietnamese Dataset Builder", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Step 0: Global Paths & Settings
        self.create_global_section(main_frame, 1)
        
        # Step 1: Download YouTube
        self.create_download_section(main_frame, 2)
        
        # Step 2: Voice Separation
        self.create_separation_section(main_frame, 3)
        
        # Step 3: Transcription
        self.create_transcription_section(main_frame, 4)
        
        # Log area
        self.create_log_section(main_frame, 5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def create_global_section(self, parent, row):
        frame = ttk.LabelFrame(parent, text="Global Paths & Settings", padding="10")
        frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Workspace Base:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.workspace_var = tk.StringVar(value=str(self.base_dir))
        ttk.Entry(frame, textvariable=self.workspace_var).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.workspace_var)).grid(row=0, column=2)

        ttk.Label(frame, text="Downloaded Audio:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10,0))
        self.download_dir_var = tk.StringVar(value=str(self.base_dir / "data" / "downloaded_audio"))
        ttk.Entry(frame, textvariable=self.download_dir_var).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10,0))
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.download_dir_var)).grid(row=1, column=2, pady=(10,0))

        ttk.Label(frame, text="Separated Voice:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.separation_output_var = tk.StringVar(value=str(self.base_dir / "data" / "separated_voice"))
        ttk.Entry(frame, textvariable=self.download_dir_var).grid(row=2, column=1, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.separation_output_var)).grid(row=2, column=2)

        ttk.Label(frame, text="Dataset Output:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(10,0))
        self.dataset_output_var = tk.StringVar(value=str(self.base_dir / "data"))
        ttk.Entry(frame, textvariable=self.dataset_output_var).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=(10,0))
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.dataset_output_var)).grid(row=3, column=2, pady=(10,0))

    def create_download_section(self, parent, row):
        frame = ttk.LabelFrame(parent, text="Step 1: Download YouTube Audio", padding="10")
        frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        # Single URL download
        ttk.Label(frame, text="YouTube URL:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.url_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.url_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.download_btn = ttk.Button(frame, text="Download", command=self.download_video)
        self.download_btn.grid(row=0, column=2)
        self.download_progress = ttk.Progressbar(frame, mode='indeterminate')

        # Batch download
        ttk.Label(frame, text="Batch URLs (one per line):").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10,0))
        self.batch_text = scrolledtext.ScrolledText(frame, height=4, width=50)
        self.batch_text.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10,0))
        self.batch_btn = ttk.Button(frame, text="Download Batch", command=self.download_batch)
        self.batch_btn.grid(row=1, column=2, pady=(10,0))
        self.batch_progress = ttk.Progressbar(frame, mode='indeterminate')

    def create_separation_section(self, parent, row):
        frame = ttk.LabelFrame(parent, text="Step 2: Separate Voice from Music", padding="10")
        frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.separation_input_var = tk.StringVar(value=str(self.base_dir / "data" / "downloaded_audio"))
        ttk.Entry(frame, textvariable=self.separation_input_var).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.separation_input_var)).grid(row=0, column=2)

        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10,0))
        ttk.Entry(frame, textvariable=self.separation_output_var).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10,0))
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.separation_output_var)).grid(row=1, column=2, pady=(10,0))

        self.separate_btn = ttk.Button(frame, text="Separate All Files", command=self.separate_all_files)
        self.separate_btn.grid(row=2, column=1, pady=(10,0))
        self.separate_progress = ttk.Progressbar(frame, mode='indeterminate')

    def create_transcription_section(self, parent, row):
        frame = ttk.LabelFrame(parent, text="Step 3: Create VITS Dataset", padding="10")
        frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.transcription_input_var = tk.StringVar(value=str(self.base_dir / "data" / "separated_voice"))
        ttk.Entry(frame, textvariable=self.transcription_input_var).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.transcription_input_var)).grid(row=0, column=2)

        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10,0))
        ttk.Entry(frame, textvariable=self.dataset_output_var).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10,0))
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.dataset_output_var)).grid(row=1, column=2, pady=(10,0))

        # Segmentation parameters
        params_frame = ttk.Frame(frame)
        params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10,0))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)

        ttk.Label(params_frame, text="Min Length (sec):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.seg_min_var = tk.DoubleVar(value=3.0)
        ttk.Entry(params_frame, textvariable=self.seg_min_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        ttk.Label(params_frame, text="Max Length (sec):").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.seg_max_var = tk.DoubleVar(value=8.0)
        ttk.Entry(params_frame, textvariable=self.seg_max_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=(0, 20))

        ttk.Label(params_frame, text="Merge Silence (sec):").grid(row=0, column=4, sticky=tk.W, padx=(0, 10))
        self.merge_sil_var = tk.DoubleVar(value=0.6)
        ttk.Entry(params_frame, textvariable=self.merge_sil_var, width=10).grid(row=0, column=5, sticky=tk.W)

        # Options
        options_frame = ttk.Frame(frame)
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10,0))

        self.append_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Append to existing dataset", 
                       variable=self.append_var).grid(row=0, column=0, sticky=tk.W)

        # Buttons
        buttons_frame = ttk.Frame(frame)
        buttons_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10,0))

        self.transcribe_btn = ttk.Button(buttons_frame, text="Create Dataset", command=self.create_dataset)
        self.transcribe_btn.grid(row=0, column=0, padx=(0, 10))

        self.inc_btn = ttk.Button(buttons_frame, text="Incremental Update", command=self.incremental_update)
        self.inc_btn.grid(row=0, column=1)

        self.transcribe_progress = ttk.Progressbar(frame, mode='indeterminate')
        self.inc_progress = ttk.Progressbar(frame, mode='indeterminate')

    def create_log_section(self, parent, row):
        frame = ttk.LabelFrame(parent, text="Log Output", padding="10")
        frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Log controls
        controls_frame = ttk.Frame(frame)
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        controls_frame.columnconfigure(0, weight=1)

        ttk.Button(controls_frame, text="Clear Log", command=self.clear_log).grid(row=0, column=0, sticky=tk.W)

        # Log text area
        self.log_text = scrolledtext.ScrolledText(frame, height=15, width=100)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def setup_logging(self):
        """Setup logging thread"""
        def log_worker():
            while True:
                try:
                    message = self.log_queue.get(timeout=0.1)
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.update_idletasks()
                except queue.Empty:
                    continue
        
        threading.Thread(target=log_worker, daemon=True).start()

    def log(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")
    
    def clear_log(self):
        """Clear log text area"""
        self.log_text.delete(1.0, tk.END)
    
    def check_directories(self):
        """Check and create necessary directories"""
        dirs = [
            "./data/downloaded_audio", 
            "./data/separated_voice", 
            "./data/wavs_16khz"  # Updated to match new format
        ]
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
        self.log("Directories checked and created")
    
    def browse_directory(self, var):
        """Browse for directory"""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
    
    def download_video(self):
        """Download single video"""
        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
        
        self.download_btn.config(state="disabled")
        self.status_var.set("Downloading...")
        self.download_progress.start()
        
        def download_thread():
            try:
                self.log(f"Starting download: {url}")
                downloader = YouTubeDownloader(self.download_dir_var.get())
                downloader.download_multiple([url])
                self.log("✅ Download completed successfully")
                self.status_var.set("Download completed")
            except Exception as e:
                self.log(f"❌ Download failed: {e}")
                self.status_var.set("Download failed")
            finally:
                self.download_btn.config(state="normal")
                self.download_progress.stop()
        
        threading.Thread(target=download_thread, daemon=True).start()
    
    def download_batch(self):
        """Download multiple videos"""
        urls_text = self.batch_text.get(1.0, tk.END).strip()
        if not urls_text:
            messagebox.showerror("Error", "Please enter URLs in the batch text area")
            return
        
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        if not urls:
            messagebox.showerror("Error", "No valid URLs found")
            return
        
        self.download_btn.config(state="disabled")
        self.status_var.set("Downloading batch...")
        self.batch_progress.start()
        
        def download_thread():
            try:
                self.log(f"Starting batch download: {len(urls)} videos")
                downloader = YouTubeDownloader(self.download_dir_var.get())
                downloader.download_multiple(urls)
                self.log("✅ Batch download completed successfully")
                self.status_var.set("Batch download completed")
            except Exception as e:
                self.log(f"❌ Batch download failed: {e}")
                self.status_var.set("Batch download failed")
            finally:
                self.download_btn.config(state="normal")
                self.batch_progress.stop()
        
        threading.Thread(target=download_thread, daemon=True).start()
    
    def separate_all_files(self):
        """Separate voice from all files"""
        self.separate_btn.config(state="disabled")
        self.status_var.set("Separating voice...")
        self.separate_progress.start()
        
        def separate_thread():
            try:
                self.log("Starting voice separation...")
                separator = VoiceSeparator(self.separation_input_var.get(), 
                                        self.separation_output_var.get())
                separator.separate_all_files()
                self.log("✅ Voice separation completed successfully")
                self.status_var.set("Voice separation completed")
            except Exception as e:
                self.log(f"❌ Voice separation failed: {e}")
                self.status_var.set("Voice separation failed")
            finally:
                self.separate_btn.config(state="normal")
                self.separate_progress.stop()
        
        threading.Thread(target=separate_thread, daemon=True).start()
    
    def create_dataset(self):
        """Create VITS dataset"""
        self.transcribe_btn.config(state="disabled")
        self.status_var.set("Creating dataset...")
        self.transcribe_progress.start()
        
        def transcribe_thread():
            try:
                self.log("Starting transcription and dataset creation...")
                transcriber = TranscriptionProcessor(
                    self.transcription_input_var.get(), 
                    self.dataset_output_var.get(),  # Use dataset_output_var instead
                    seg_min_len_sec=self.seg_min_var.get(),
                    seg_max_len_sec=self.seg_max_var.get(),
                    seg_top_db=30.0,
                    seg_merge_silence_sec=self.merge_sil_var.get(),
                    append=self.append_var.get(),
                )
                transcriber.process_all_files()
                # Note: create_dataset_splits is now handled automatically in process_all_files
                self.log("✅ Dataset created successfully")
                self.status_var.set("Dataset created")
            except Exception as e:
                self.log(f"❌ Dataset creation failed: {e}")
                self.status_var.set("Dataset creation failed")
            finally:
                self.transcribe_btn.config(state="normal")
                self.transcribe_progress.stop()
        
        threading.Thread(target=transcribe_thread, daemon=True).start()

    def incremental_update(self):
        """Incremental: separate new downloads, append & split"""
        self.status_var.set("Incremental update...")
        self.inc_progress.start()
        def run_update():
            try:
                download_dir = self.download_dir_var.get()
                separated_dir = self.separation_output_var.get()
                dataset_out = self.dataset_output_var.get()
                self.log("Scanning for new downloads...")
                created = separate_new(Path(download_dir), Path(separated_dir))
                if not created:
                    self.log("No new files to separate. Nothing to append.")
                else:
                    self.log(f"Appending {len(created)} separated files and re-splitting...")
                    append_transcribe_and_split(Path(separated_dir), Path(dataset_out), created)
                    self.log("✅ Incremental update done")
                self.status_var.set("Incremental update completed")
            except Exception as e:
                self.log(f"❌ Incremental update failed: {e}")
                self.status_var.set("Incremental update failed")
            finally:
                self.inc_progress.stop()
        threading.Thread(target=run_update, daemon=True).start()


def main():
    root = tk.Tk()
    app = YouTubeTTSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 