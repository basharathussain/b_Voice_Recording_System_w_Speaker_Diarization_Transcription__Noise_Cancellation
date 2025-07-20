
# 📢 Enhanced Voice Recording System with Speaker Diarization, Transcription, and Noise Cancellation

## 1. Overview
This system delivers **real-time, multilingual, multi-speaker transcription** with **speaker diarization** and **voice print registration**, enhanced by **DeepFilterNet** for **noise cancellation**. It supports **Urdu, English, and mixed-language** conversations and is optimized for use in noisy or uncontrolled environments.

## 2. Core System Architecture
```
Audio Input
   ↓
DeepFilterNet (Noise Removal)
   ↓
Voice Activity Detection (VAD)
   ↓
pyannote-audio (Speaker Diarization & Embeddings)
   ↓
Whisper (Transcription + Language Detection)
   ↓
Real-time Output (JSON / WebSocket / REST API)
```

## 3. Primary Components
| Component               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Whisper large-v3**   | Multilingual speech recognition (Urdu + English + code-switching)           |
| **pyannote-audio 3.x** | Real-time speaker diarization & 512-D embedding extraction                  |
| **DeepFilterNet2**     | Neural noise cancellation for clean, enhanced audio                         |
| **Voice Print DB**     | PostgreSQL/SQLite for storing registered speaker embeddings                 |
| **Real-time Buffering**| Threaded audio buffer processing with sub-second latency                    |

## 4. Voice Print Registration Module

### 🛂 Registration Workflow
1. Record 30–60s clean enrollment audio.
2. Extract multiple speaker embeddings (512-D) using pyannote.
3. Average embeddings to create voice print.
4. Store with speaker ID, metadata in SQL DB.
5. Perform verification using a test utterance.

### ✅ Requirements
- Sample Rate: ≥ 16kHz
- Format: 16-bit PCM WAV
- Environment: Minimal background noise
- Matching: Cosine similarity threshold (0.75–0.85)

## 5. Real-time Audio Processing Pipeline

### 🎛️ Audio Parameters
| Parameter       | Value                           |
|----------------|----------------------------------|
| Buffer Size     | 1–3 sec (sliding window)         |
| Overlap         | 0.5–1 sec                        |
| Sample Rate     | 16kHz                            |
| Format          | 16-bit PCM                       |

### 🔊 Pipeline Flow
```
Capture → Enhance → Diarize → Transcribe → Match → Stream Output
```

## 6. Diarization & Speaker Matching

### 🔍 pyannote-audio Workflow
- **Segmentation**: Speaker turns detected
- **Embedding**: 512-D vector per segment
- **Matching**: Against registered embeddings (cosine similarity)
- **Clustering**: Unknown speakers labeled `Speaker_1`, `Speaker_2`, etc.
- **Temporal Smoothing**: Reduces rapid speaker switches

## 7. Transcription and Language Handling

| Feature                 | Description                                                     |
|------------------------|-----------------------------------------------------------------|
| **Language Detection** | Urdu, English, and mixed-language per segment                   |
| **Code-switching**     | Seamlessly handled within the same utterance                    |
| **Output**             | Speaker-tagged JSON with timestamps, language labels, and text  |

## 8. Noise Cancellation Integration (DeepFilterNet2)

### 🔧 Technical Pipeline
```
Raw Audio → DeepFilterNet → VAD → pyannote → Whisper
```

### 🧠 DeepFilterNet Capabilities
| Removes                          | Preserves                                  |
|----------------------------------|---------------------------------------------|
| Fan, traffic, chatter, reverb    | Voice clarity, identity, tone, multi-speaker|
| Stationary/non-stationary noise | Emotional prosody                          |

### ⏱️ Added Latency: 20–40ms

## 9. Technical Specifications

### 🖥️ Hardware Requirements
| Component        | Specification                        |
|------------------|--------------------------------------|
| CPU              | 8+ cores (e.g., Intel i7/Ryzen 7)     |
| RAM              | 20–32 GB                              |
| GPU              | RTX 3070/4060 or higher, 10GB+ VRAM   |
| Storage          | SSD, 100GB+                           |
| Audio Interface  | Low-noise input, professional grade   |

### 🧑‍💻 Software Stack
- Python 3.9+
- PyTorch 2.0+ w/ CUDA
- `faster-whisper`
- `pyannote-audio`
- `deepfilternet`
- `torch-audio`, `librosa`, `soundfile`
- `asyncio`, `threading` for real-time handling
- WebRTC VAD

## 10. Memory & Model Footprint
| Model                | VRAM Usage           |
|----------------------|----------------------|
| Whisper large-v3     | ~3GB                 |
| pyannote-audio       | ~2GB                 |
| DeepFilterNet2       | ~1–2GB               |
| Speaker Embeddings   | ~1MB / 1000 speakers |
| **Total VRAM Needed**| ~8–10GB              |

## 11. Multi-speaker Support

| Attribute                | Value                             |
|--------------------------|-----------------------------------|
| Max Concurrent Speakers  | 6–8 (ideal: 2–4)                  |
| Active Monitoring        | Up to 50 known speakers/session   |
| Registration Capacity    | Unlimited                         |
| Unknown Speaker Clustering | Auto-labeled `Speaker_n`         |
| Overlap Handling         | Partial; accuracy drops above 4   |

## 12. Language & Output Handling

| Urdu Support            | Whisper large-v3 (fine-tuned or default multilingual) |
|-------------------------|--------------------------------------------------------|
| Script Output           | Native Urdu + optional Roman Urdu                      |
| Code-switching Handling | Per segment and speaker turn                           |

### 🧾 Output Format (JSON Stream)
```json
{
  "timestamp": "2025-07-20T10:30:45.123Z",
  "start_time": 125.4,
  "end_time": 128.7,
  "speaker_id": "john_doe",
  "speaker_confidence": 0.89,
  "text": "یہ اردو اور English mixed sentence ہے",
  "language": "mixed",
  "confidence": 0.92
}
```

## 13. Integration APIs

- **WebSocket**: Real-time transcription stream
- **REST API**: Registration, session start/stop, speaker info
- **Webhook**: Event-based triggers for transcription chunks
- **File Export**: `.srt`, `.vtt`, `.json`

## 14. Performance Targets

### ⚡ Latency
| Stage              | Delay         |
|--------------------|---------------|
| DeepFilterNet      | 20–40ms       |
| Diarization        | 2–4 sec       |
| Transcription      | 1–3 sec       |
| Voice Matching     | <100ms        |
| **Total**          | ~3.5–7.5 sec  |

### 🚀 Throughput
- **Realtime audio**: 1.0x speed (no lag)
- **Sessions**: 1–2 parallel streams
- **Buffering**: Circular, 30–60s retention

## 15. Accuracy Expectations

| Condition               | Diarization | Transcription (En) | Transcription (Ur) | Mixed |
|-------------------------|-------------|---------------------|---------------------|-------|
| Clean studio            | 95%+        | 95%+                | 90%+                | 85%+  |
| Office noise (DFN)      | 90–97%      | 88–93%              | 82–88%              | 78–85%|
| Outdoor noisy (DFN)     | 75–85%      | 80–85%              | 75–80%              | 70–80%|

## 16. Limitations

- Diarization drops with >4 overlapping speakers
- Code-switching introduces brief delays
- Heavy background noise reduces registration accuracy (without DeepFilterNet)
- Initial cold start time: 10–15 seconds

## 17. System Configuration & Tuning

### 🎚️ DeepFilterNet Settings

| Profile          | Enhancement Strength |
|------------------|----------------------|
| Studio           | 0.3 (light)          |
| Office           | 0.7 (medium)         |
| Street/Outdoor   | 0.9 (strong)         |
| Conference Room  | 0.8 (medium-high)    |

## 18. Implementation Considerations

| Topic                 | Strategy                                                      |
|-----------------------|---------------------------------------------------------------|
| Memory Management     | Circular buffers, efficient VRAM sharing                      |
| Threading             | Separate threads for audio I/O, enhancement, processing       |
| Performance Scaling   | Model simplification, load balancing                          |
| Failure Handling      | Fallback to raw audio if enhancement fails                    |

## ✅ Summary
This enhanced system is ideal for:
- Real-time meeting transcriptions
- Call center recordings
- Podcasts and panel discussions
- Multilingual transcription with diarization

With **DeepFilterNet noise cancellation**, **Whisper transcription**, and **pyannote speaker tracking**, the system is production-grade, robust under noisy and dynamic conditions, and scalable for multiple users and languages.
