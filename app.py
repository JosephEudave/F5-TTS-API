from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback
from pathlib import Path
import torch
from f5_tts.api import F5TTS
import tomli
import soundfile as sf
import datetime
import asyncio
import sounddevice as sd
import numpy as np
import os
import sys
import codecs
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

# Force UTF-8 encoding for stdout and stderr
if sys.platform == 'win32':
    # Python UTF-8 Mode
    if hasattr(sys, 'set_utf8_mode'):
        sys.set_utf8_mode(True)
    
    # Force console to use UTF-8
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)

# Create a custom StreamHandler that forces UTF-8
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if stream is None:
            if sys.platform == 'win32':
                # On Windows, use a custom stream that handles UTF-8 properly
                self.stream = codecs.getwriter('utf-8')(sys.stdout.buffer)
            else:
                self.stream = sys.stdout
        else:
            self.stream = stream

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Ensure the message is properly encoded and handle emoji characters
            if isinstance(msg, str):
                # Replace problematic characters with ASCII alternatives
                msg = msg.replace('➡️', '->')
                msg = msg.encode('utf-8', errors='replace').decode('utf-8')
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Set up logging with UTF-8 encoding
os.makedirs('logs', exist_ok=True)  # Ensure logs directory exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8StreamHandler(),
        logging.FileHandler('logs/f5_tts_api.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger("f5_tts")

# Helper function for safe logging
def safe_log(logger, level, message):
    """Safely log messages by replacing problematic characters"""
    if isinstance(message, str):
        message = message.replace('➡️', '->')
    getattr(logger, level)(message)
    # Also print to console for immediate visibility
    print(f"[{level.upper()}] {message}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize TTS model
logger.info("Initializing TTS model (F5TTS_Base)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name()}")

try:
    tts_model = F5TTS(
        model='F5TTS_Base',
        hf_cache_dir=os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'),  # Explicit cache dir
        device=device  # Explicitly pass the device
    )
    logger.info("F5TTS_Base model loaded successfully")
    logger.info("Model configuration loaded and verified")
    # Add device verification logging
    logger.info(f"Model device: {next(tts_model.ema_model.parameters()).device}")
    logger.info(f"Vocoder device: {next(tts_model.vocoder.parameters()).device}")
    # Add more detailed device info
    if torch.cuda.is_available():
        logger.info(f"Model memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"Model memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
except Exception as e:
    logger.error(f"Failed to load F5TTS_Base model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Load custom configuration
try:
    with open('custom.toml', 'rb') as f:  # tomli requires binary mode
        custom_config = tomli.load(f)
    logger.info("Successfully loaded custom configuration")
except Exception as e:
    logger.error(f"Error loading custom configuration: {str(e)}")
    raise

class ConnectionState(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

@dataclass
class ConnectionInfo:
    state: ConnectionState
    last_error: Optional[str] = None
    reconnect_attempts: int = 0
    current_task: Optional[asyncio.Task] = None
    audio_playing: bool = False

class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1  # seconds
        self.playback_status: Dict[str, bool] = {}  # Track playback status for each client

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_info[client_id] = ConnectionInfo(state=ConnectionState.CONNECTED)
        self.playback_status[client_id] = False
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_info:
            del self.connection_info[client_id]
        if client_id in self.playback_status:
            del self.playback_status[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def handle_connection(self, websocket: WebSocket, client_id: str):
        try:
            await self.connect(websocket, client_id)
            while True:
                try:
                    message = await websocket.receive_text()
                    if message == "waiting" and self.playback_status.get(client_id, False):
                        await websocket.send_json({
                            "type": "status",
                            "message": "still_playing"
                        })
                    else:
                        await self.handle_message(websocket, client_id, message)
                except WebSocketDisconnect:
                    await self.handle_disconnect(client_id)
                    break
                except Exception as e:
                    await self.handle_error(client_id, str(e))
        finally:
            self.disconnect(client_id)

    async def handle_disconnect(self, client_id: str):
        info = self.connection_info.get(client_id)
        if info:
            info.state = ConnectionState.DISCONNECTED
            if info.current_task and not info.current_task.done():
                info.current_task.cancel()
            await self.attempt_reconnect(client_id)

    async def attempt_reconnect(self, client_id: str):
        info = self.connection_info.get(client_id)
        if not info:
            return

        info.state = ConnectionState.RECONNECTING
        info.reconnect_attempts += 1

        if info.reconnect_attempts > self.max_reconnect_attempts:
            info.state = ConnectionState.ERROR
            logger.error(f"Max reconnection attempts reached for client {client_id}")
            return

        try:
            # Wait before attempting to reconnect
            await asyncio.sleep(self.reconnect_delay)
            
            # Instead of creating a new WebSocket, we'll just update the state
            # The client should handle reconnection on their end
            info.state = ConnectionState.DISCONNECTED
            logger.info(f"Client {client_id} should reconnect from their end")
            
        except Exception as e:
            await self.handle_error(client_id, f"Reconnection failed: {str(e)}")

    async def handle_error(self, client_id: str, error: str):
        info = self.connection_info.get(client_id)
        if info:
            info.state = ConnectionState.ERROR
            info.last_error = error
            logger.error(f"Error for client {client_id}: {error}")
            await self.attempt_reconnect(client_id)

    async def handle_message(self, websocket: WebSocket, client_id: str, message: str):
        info = self.connection_info.get(client_id)
        if not info:
            return

        if message == "start":
            await websocket.send_text("ready")
            raw_text = await websocket.receive_text()
            text = raw_text.replace('{"text": "', '').replace('"}', '').strip()
            
            # Create a new task for audio processing
            info.current_task = asyncio.create_task(
                self.process_audio(websocket, client_id, text)
            )

    async def process_audio(self, websocket: WebSocket, client_id: str, text: str):
        info = self.connection_info.get(client_id)
        if not info:
            return

        try:
            info.audio_playing = True
            self.playback_status[client_id] = True
            
            # Update config with the received text
            inference_config = custom_config.copy()
            inference_config['gen_text'] = text
            
            # Generate speech
            wav, sr, spec = tts_model.infer(
                ref_file=inference_config.get('ref_audio', ''),
                ref_text=inference_config.get('ref_text', ''),
                gen_text=text,
            )

            # Create an event to signal when playback is complete
            playback_complete = asyncio.Event()
            
            # Convert wav to list for easier manipulation
            wav_list = wav.tolist()
            current_position = 0

            def callback(outdata, frames, time, status):
                nonlocal current_position
                if status:
                    print(f"Audio callback status: {status}")
                
                if current_position >= len(wav_list):
                    outdata.fill(0)
                    if not playback_complete.is_set():
                        playback_complete.set()
                    return
                
                # Calculate how many samples we can send
                remaining = len(wav_list) - current_position
                samples_to_send = min(frames, remaining)
                
                # Fill the output buffer
                outdata[:samples_to_send, 0] = wav_list[current_position:current_position + samples_to_send]
                if samples_to_send < frames:
                    outdata[samples_to_send:, 0] = 0
                
                current_position += samples_to_send
                
                if current_position >= len(wav_list):
                    if not playback_complete.is_set():
                        playback_complete.set()

            # Play audio using sounddevice with callback
            try:
                stream = sd.OutputStream(
                    samplerate=sr,
                    channels=1,
                    dtype=np.float32,
                    callback=callback
                )
                with stream:
                    # Send talking message to client right when audio starts playing
                    await websocket.send_json({
                        "type": "status",
                        "message": "talking"
                    })
                    logger.info(f"Sent 'talking' message to client {client_id}")
                    
                    await playback_complete.wait()  # Wait for playback to complete
                    # Add a small delay to ensure audio is fully played
                    await asyncio.sleep(0.1)
            except Exception as audio_error:
                error_msg = f"Audio playback error: {str(audio_error)}"
                print(f"\n[ERROR] {error_msg}")
                logger.error(error_msg)
                raise
            finally:
                info.audio_playing = False
                self.playback_status[client_id] = False
                # Ensure we send the done message
                try:
                    await websocket.send_json({
                        "type": "status",
                        "message": "done"
                    })
                    logger.info(f"Sent 'done' message to client {client_id}")
                except Exception as e:
                    logger.error(f"Failed to send 'done' message to client {client_id}: {str(e)}")

        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            print(f"\n[ERROR] Traceback:\n{traceback.format_exc()}")
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            info.audio_playing = False
            self.playback_status[client_id] = False
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except Exception as send_error:
                logger.error(f"Failed to send error message to client {client_id}: {str(send_error)}")

# Initialize the connection manager
connection_manager = WebSocketConnectionManager()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_device": device,
        "model_loaded": tts_model is not None
    }

@app.websocket("/tts")
async def websocket_tts(websocket: WebSocket):
    client_id = str(id(websocket))
    await connection_manager.handle_connection(websocket, client_id) 