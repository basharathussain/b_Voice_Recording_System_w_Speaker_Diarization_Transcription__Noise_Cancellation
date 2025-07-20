# b_Voice_Recording_System_w_Speaker_Diarization_Transcription__Noise_Cancellation

voice-system/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Configuration settings
│   ├── models.py                  # Pydantic models & DB schemas
│   ├── database.py                # Speaker profile database
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── audio_input.py         # Audio capture & streaming
│   │   ├── audio_buffer.py        # Circular buffer management
│   │   └── audio_utils.py         # Audio processing utilities
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── deepfilternet_worker.py # Noise cancellation
│   │   ├── diarization_worker.py   # Speaker diarization
│   │   ├── whisper_worker.py       # Speech transcription
│   │   └── voice_print_worker.py   # Speaker registration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py              # API endpoints
│   │   └── websocket.py           # Real-time WebSocket
│   └── utils/
│       ├── __init__.py
│       ├── redis_client.py        # Redis queue management
│       └── logging_config.py      # Logging setup
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md

