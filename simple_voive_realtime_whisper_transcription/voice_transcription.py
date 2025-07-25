#!/usr/bin/env python3
"""
Real-time Voice Transcription using Whisper
Configurable models, console output, and file saving
"""

import os
import time
import json
import threading
import queue
from datetime import datetime
from pathlib import Path
import sys
import contextlib
from io import StringIO

# Set environment variables BEFORE importing anything else
os.environ["TQDM_DISABLE"] = "1"
os.environ["TQDM_MINITERS"] = "1"
os.environ["TQDM_MININTERVAL"] = "0"

import pyaudio
import numpy as np

# Monkey patch tqdm BEFORE importing whisper
import tqdm
import tqdm.auto

class DummyTqdm:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, *args, **kwargs):
        pass
    def close(self):
        pass
    def set_description(self, *args, **kwargs):
        pass
    def set_postfix(self, *args, **kwargs):
        pass
    def write(self, *args, **kwargs):
        pass
    def __iter__(self):
        return iter([])
    def __len__(self):
        return 0

# Replace all tqdm instances
tqdm.tqdm = DummyTqdm
tqdm.auto.tqdm = DummyTqdm

import whisper
from scipy.io import wavfile
import tempfile
import torch

# Suppress Whisper warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class RealTimeTranscriber:
    def __init__(self, model_name="base", language="auto", chunk_duration=3, device="auto"):
        """
        Initialize the real-time transcriber
        
        Args:
            model_name (str): Whisper model size ("tiny", "base", "small", "medium", "large")
            language (str): Language code ("en", "ur", "auto" for auto-detection)
            chunk_duration (int): Audio chunk duration in seconds
            device (str): Device to run on ("cpu", "gpu", "auto")
        """
        self.model_name = model_name
        self.language = None if language == "auto" else language
        self.chunk_duration = chunk_duration
        
        # Set device
        self.device = self._setup_device(device)
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Initialize queues and flags
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.running = False
        
        # Duplicate detection
        self.last_transcription = ""
        self.last_transcription_time = 0
        
        # Load Whisper model with device specification
        print(f"Loading Whisper model '{model_name}' on {self.device}...")
        
        # Suppress output during model loading
        with contextlib.redirect_stdout(StringIO()), \
             contextlib.redirect_stderr(StringIO()):
            self.model = whisper.load_model(model_name, device=self.device)
            
        print(f"Model loaded successfully on {self.device}!")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Create output directory
        self.output_dir = Path("transcriptions")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create session file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"transcription_{timestamp}.txt"
        self.json_file = self.output_dir / f"transcription_{timestamp}.json"
        
        # Initialize transcription log
        self.transcription_log = []
        
        print(f"Transcriptions will be saved to: {self.output_file}")
        print(f"JSON log will be saved to: {self.json_file}")

    def _setup_device(self, device_choice):
        """Setup and return the appropriate device"""
        if device_choice.lower() == "cpu":
            print("Using CPU for processing")
            return "cpu"
        elif device_choice.lower() == "gpu":
            if torch.cuda.is_available():
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                return "cuda"
            else:
                print("GPU requested but CUDA not available. Falling back to CPU.")
                return "cpu"
        else:  # auto
            if torch.cuda.is_available():
                print(f"Auto-detected GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                return "cuda"
            else:
                print("Auto-detected: Using CPU (CUDA not available)")
                return "cpu"

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.running:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_queue.put(audio_data.copy())
        return (in_data, pyaudio.paContinue)

    def start_audio_stream(self):
        """Start the audio input stream"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print("Audio stream started")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            raise

    def process_audio(self):
        """Process audio chunks and transcribe them"""
        audio_buffer = []
        last_transcription = ""
        last_transcription_time = 0
        
        while self.running:
            try:
                # Get audio chunk with timeout
                chunk = self.audio_queue.get(timeout=1.0)
                audio_buffer.extend(chunk)
                
                # Process when we have enough audio
                if len(audio_buffer) >= self.chunk_size:
                    # Convert to numpy array and normalize
                    audio_array = np.array(audio_buffer[:self.chunk_size], dtype=np.float32)
                    audio_array = audio_array / 32768.0  # Normalize int16 to float32
                    
                    # Clear processed audio from buffer (no overlap to prevent duplicates)
                    audio_buffer = audio_buffer[self.chunk_size:]
                    
                    # Transcribe
                    current_text = self.transcribe_chunk(audio_array, self.last_transcription, self.last_transcription_time)
                    if current_text:
                        self.last_transcription = current_text
                        self.last_transcription_time = time.time()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")

    def transcribe_chunk(self, audio_array, last_transcription="", last_time=0):
        """Transcribe an audio chunk with duplicate detection"""
        try:
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                wavfile.write(temp_file.name, self.sample_rate, 
                            (audio_array * 32767).astype(np.int16))
                temp_path = temp_file.name
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                temp_path,
                language=self.language,
                fp16=False,
                verbose=False
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Extract text and metadata
            text = result.get("text", "").strip()
            language = result.get("language", "unknown")
            
            # Skip if text is empty or too similar to last transcription
            if not text:
                return None
            
            # Simple duplicate detection using similarity
            current_time = time.time()
            if self._is_duplicate_transcription(text, last_transcription, current_time, last_time):
                return None
            
            timestamp = datetime.now()
            
            # Create transcription entry
            entry = {
                "timestamp": timestamp.isoformat(),
                "text": text,
                "language": language,
                "model": self.model_name
            }
            
            # Add to log
            self.transcription_log.append(entry)
            
            # Display in console
            time_str = timestamp.strftime("%H:%M:%S")
            print(f"[{time_str}] ({language}): {text}")
            
            # Save to file immediately
            self.save_transcription(entry)
            
            # Return the text for duplicate tracking
            return text
            
        except Exception as e:
            print(f"Error transcribing chunk: {e}")
            return None

    def _is_duplicate_transcription(self, current_text, last_text, current_time, last_time):
        """Check if current transcription is a duplicate of the last one"""
        # If too much time has passed, it's not a duplicate
        if current_time - last_time > self.chunk_duration * 2:
            return False
        
        # If texts are identical, it's a duplicate
        if current_text.lower().strip() == last_text.lower().strip():
            return True
        
        # Check for high similarity (simple word overlap check)
        current_words = set(current_text.lower().split())
        last_words = set(last_text.lower().split())
        
        if not current_words or not last_words:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(current_words.intersection(last_words))
        union = len(current_words.union(last_words))
        similarity = intersection / union if union > 0 else 0
        
        # If similarity is very high (>80%), consider it a duplicate
        return similarity > 0.8

    def save_transcription(self, entry):
        """Save transcription to text and JSON files"""
        try:
            # Append to text file
            with open(self.output_file, "a", encoding="utf-8") as f:
                timestamp = entry["timestamp"]
                text = entry["text"]
                language = entry["language"]
                f.write(f"[{timestamp}] ({language}): {text}\n")
            
            # Save complete JSON log
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(self.transcription_log, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving transcription: {e}")

    def start(self):
        """Start the real-time transcription"""
        print("\n" + "="*60)
        print("REAL-TIME VOICE TRANSCRIPTION")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Language: {self.language or 'auto-detect'}")
        print(f"Chunk duration: {self.chunk_duration}s")
        if self.device == "cuda":
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("Press Ctrl+C to stop")
        print("="*60)
        
        self.running = True
        
        try:
            # Start audio stream
            self.start_audio_stream()
            
            # Start processing thread
            process_thread = threading.Thread(target=self.process_audio, daemon=True)
            process_thread.start()
            
            print("\nListening... Speak into your microphone!")
            print("(There may be a 3-5 second delay for processing)\n")
            
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nStopping transcription...")
            self.stop()
        except Exception as e:
            print(f"\nError: {e}")
            self.stop()

    def stop(self):
        """Stop the transcription"""
        self.running = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        
        # Final save
        if self.transcription_log:
            print(f"\nFinal transcription saved to:")
            print(f"  Text: {self.output_file}")
            print(f"  JSON: {self.json_file}")
            print(f"Total transcribed segments: {len(self.transcription_log)}")

def main():
    """Main function with configuration options"""
    print("Real-time Voice Transcription with Whisper")
    print("=" * 50)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU detected - will use CPU")
    
    # Configuration options
    models = ["tiny", "base", "small", "medium", "large"]
    languages = ["auto", "en", "ur", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
    devices = ["auto", "gpu", "cpu"]
    
    print("\nAvailable models (accuracy vs speed):")
    model_info = {
        "tiny": "~1GB VRAM, fastest, lower accuracy",
        "base": "~1GB VRAM, good balance",
        "small": "~2GB VRAM, better accuracy",
        "medium": "~5GB VRAM, high accuracy",
        "large": "~10GB VRAM, highest accuracy"
    }
    for i, model in enumerate(models):
        print(f"  {i+1}. {model:<8} - {model_info[model]}")
    
    print("\nDevice options:")
    print("  1. auto - Automatically detect best device")
    print("  2. gpu  - Force GPU usage (requires CUDA)")
    print("  3. cpu  - Force CPU usage")
    
    # Get user preferences
    try:
        model_choice = input(f"\nSelect model (1-{len(models)}) [default: 2 (base)]: ").strip()
        if not model_choice:
            model_choice = "2"
        model_name = models[int(model_choice) - 1]
    except (ValueError, IndexError):
        model_name = "base"
        print(f"Using default model: {model_name}")
    
    try:
        device_choice = input("Select device (1-3) [default: 1 (auto)]: ").strip()
        if not device_choice:
            device_choice = "1"
        device = devices[int(device_choice) - 1]
    except (ValueError, IndexError):
        device = "auto"
        print(f"Using default device: {device}")
    
    try:
        language = input("Enter language code (e.g., 'en', 'ur', 'auto') [default: auto]: ").strip()
        if not language:
            language = "auto"
    except:
        language = "auto"
    
    try:
        chunk_duration = input("Enter chunk duration in seconds [default: 3]: ").strip()
        if not chunk_duration:
            chunk_duration = 3
        else:
            chunk_duration = int(chunk_duration)
    except:
        chunk_duration = 3
    
    # Create and start transcriber
    transcriber = RealTimeTranscriber(
        model_name=model_name,
        language=language,
        chunk_duration=chunk_duration,
        device=device
    )
    
    transcriber.start()

if __name__ == "__main__":
    main()