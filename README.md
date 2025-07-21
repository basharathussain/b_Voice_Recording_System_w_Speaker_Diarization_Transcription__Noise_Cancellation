# b_Voice_Recording_System_w_Speaker_Diarization_Transcription__Noise_Cancellation

<img width="748" height="760" alt="image" src="https://github.com/user-attachments/assets/e8775c6f-805f-4fea-9d03-8af2518aacdb" />


# Voice Recording System

A real-time voice recording system with speaker diarization, speech transcription, and noise cancellation capabilities. Supports English, Urdu, and mixed-language processing.

## Features

- **Real-time Audio Processing**: Live audio capture and processing
- **Speaker Diarization**: Identify and separate multiple speakers using pyannote-audio
- **Speech Transcription**: Convert speech to text using OpenAI Whisper
- **Noise Cancellation**: Real-time audio enhancement using DeepFilterNet
- **Voice Print Registration**: Register and identify known speakers
- **Multi-language Support**: English, Urdu, and mixed-language detection
- **WebSocket API**: Real-time communication for live transcription
- **REST API**: Complete API for system management
- **Database Storage**: Persistent storage of sessions and speaker profiles

## Architecture

The system uses a microservice architecture with separate workers for each AI processing task:

```
Audio Input → DeepFilterNet (Noise Cancellation) → pyannote (Diarization) → Whisper (Transcription) → Output
```

### Core Components

1. **DeepFilterNetWorker**: Real-time noise cancellation
2. **DiarizationWorker**: Speaker identification and segmentation
3. **WhisperWorker**: Speech-to-text transcription
4. **VoicePrintWorker**: Speaker registration and verification
5. **FastAPI Application**: REST API and WebSocket endpoints
6. **PostgreSQL Database**: Data persistence
7. **Redis**: Queue management and caching

## Prerequisites

### Hardware Requirements

- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA RTX 3070/4060 or better (8GB+ VRAM)
- **Storage**: SSD with 100GB+ free space
- **Audio Interface**: Professional audio interface for clean input

### Software Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL 15+
- Redis 7+

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd voice-system

python -m venv .venv
.\.venv\Scripts\activate

```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .venv

# Edit .venv with your settings
nano .venv
```

**Important**: Add your Hugging Face access token to `.venv -> .env.example`:
```bash
PYANNOTE_ACCESS_TOKEN=your_huggingface_token_here
```

### 3. Docker Deployment (Recommended)

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f voice-api
```

### 4. Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Start PostgreSQL and Redis
# (Instructions depend on your OS)

# Run database migrations
python -c "from app.database import init_database; import asyncio; asyncio.run(init_database())"

# Start the application
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Speaker Management
- `POST /api/v1/speakers/register` - Register new speaker
- `GET /api/v1/speakers` - List all speakers
- `DELETE /api/v1/speakers/{speaker_id}` - Delete speaker

#### Session Management
- `GET /api/v1/sessions` - List recent sessions
- `GET /api/v1/sessions/{session_id}` - Get session details
- `GET /api/v1/sessions/{session_id}/transcriptions` - Get transcriptions

#### System Status
- `GET /health` - System health check
- `GET /stats` - System statistics
- `GET /api/v1/system/status` - Detailed system status

#### WebSocket
- `WS /ws/{client_id}` - Real-time audio processing

## Usage Examples

### 1. Register a Speaker

```python
import requests

# Register speaker with audio file
with open("speaker_audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/speakers/register",
        data={"speaker_id": "john_doe", "name": "John Doe"},
        files={"audio_file": f}
    )
    print(response.json())
```

### 2. WebSocket Real-time Processing

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/client123');

ws.onopen = function() {
    // Start session
    ws.send(JSON.stringify({
        type: 'start_session',
        session_id: 'session123'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'transcription') {
        console.log('Transcription:', data.data.text);
        console.log('Speaker:', data.data.speaker_id);
    }
};

// Send audio chunks (base64 encoded)
function sendAudioChunk(audioData) {
    ws.send(JSON.stringify({
        type: 'audio_chunk',
        audio_data: btoa(audioData),
        timestamp: Date.now()
    }));
}
```

### 3. Process Audio File

```python
import requests

# Upload and process audio file
with open("meeting_recording.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/audio/process",
        files={"audio_file": f},
        data={"session_id": "meeting123"}
    )
    print(response.json())
```

## Configuration

### Model Configuration

Edit worker settings in `.env`:

```bash
# Whisper settings
WHISPER_MODEL=large-v3  # Options: tiny, base, small, medium, large-v3
WHISPER_DEVICE=cuda     # Options: cuda, cpu

# Enhancement settings
ENHANCEMENT_STRENGTH=0.8      # 0.0-1.0
NOISE_REDUCTION_DB=20.0       # 0-40 dB

# Speaker settings
SPEAKER_CONFIDENCE_THRESHOLD=0.75  # 0.0-1.0
MAX_SPEAKERS=6                     # Maximum concurrent speakers
```

### Performance Tuning

For better performance:

1. **GPU Memory**: Ensure sufficient VRAM (8GB+ recommended)
2. **CPU Cores**: More cores = better parallel processing
3. **Queue Sizes**: Adjust queue sizes in worker configurations
4. **Buffer Settings**: Tune audio buffer duration for latency vs. accuracy

## Monitoring

### Health Checks

```bash
# System health
curl http://localhost:8000/health

# Detailed statistics
curl http://localhost:8000/stats
```

### Logs

```bash
# Application logs
tail -f logs/voice_system.log

# Error logs
tail -f logs/errors.log

# Performance logs
tail -f logs/performance.log
```

### Grafana Dashboard (Optional)

If using the full Docker Compose setup:
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

## Troubleshooting

### Common Issues

1. **CUDA Not Available**
   - Verify NVIDIA drivers and CUDA installation
   - Set `WHISPER_DEVICE=cpu` in `.env` for CPU-only mode

2. **Model Download Failures**
   - Check internet connection
   - Verify Hugging Face token is valid
   - Ensure sufficient disk space

3. **Audio Device Issues**
   - Check audio device permissions
   - Verify audio device availability with `GET /api/v1/audio/devices`

4. **Memory Issues**
   - Reduce number of workers
   - Use smaller Whisper model
   - Increase system RAM/VRAM

### Performance Issues

1. **High Latency**
   - Reduce buffer duration
   - Use smaller models
   - Check CPU/GPU utilization

2. **Low Accuracy**
   - Improve audio quality
   - Increase enhancement strength
   - Use larger models

## Development

### Project Structure

```
voice-system/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration
│   ├── models.py              # Pydantic models
│   ├── database.py            # Database operations
│   ├── audio/                 # Audio processing
│   ├── workers/               # AI processing workers
│   ├── api/                   # API routes and WebSocket
│   └── utils/                 # Utilities
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Adding New Features

1. **New Worker**: Create worker in `app/workers/`
2. **New API Endpoint**: Add route in `app/api/routes.py`
3. **New Model**: Add Pydantic model in `app/models.py`
4. **Database Changes**: Update `app/database.py`

### Testing

```bash
# Run tests (when implemented)
pytest tests/

# Load testing
# Use tools like Artillery or Locust for WebSocket load testing
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Create GitHub issue
- Check logs for error details
- Verify configuration settings
- Ensure all dependencies are installed correctly
