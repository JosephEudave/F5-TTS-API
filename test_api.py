import asyncio
import websockets
import json
import os
import sys
import base64
import wave
import io
import numpy as np

async def test_model(text):
    print("Starting model test...")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    output_file = "output/test_output.wav"
    
    print(f"\nGenerating speech for text: '{text}'")
    
    try:
        # Connect to the WebSocket server
        uri = "ws://localhost:9000/tts"
        async with websockets.connect(uri) as websocket:
            # Send start message
            await websocket.send("start")
            
            # Wait for ready message
            response = await websocket.recv()
            if response != "ready":
                raise Exception(f"Unexpected response: {response}")
            
            # Send the text
            await websocket.send(text)
            
            # Receive audio data
            audio_data = bytearray()
            while True:
                try:
                    chunk = await websocket.recv()
                    if chunk == b"END":
                        break
                    audio_data.extend(chunk)
                except websockets.exceptions.ConnectionClosed:
                    break
            
            # Convert received float32 data to int16
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            audio_int16 = np.int16(audio_array * 32767)
            
            # Save the audio file
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)  # F5-TTS default sample rate
                wf.writeframes(audio_int16.tobytes())
            
            print(f"\nTest successful!")
            print(f"Generated audio saved to: {output_file}")
            
            return True
            
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide text as a command line argument")
        sys.exit(1)
    
    text = sys.argv[1]
    asyncio.run(test_model(text)) 