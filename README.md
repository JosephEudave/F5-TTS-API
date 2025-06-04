# F5-TTS WebSocket API Implementation

This is a WebSocket API implementation for F5-TTS that allows other programs to access TTS services through a WebSocket interface.

## ⚠️ Important Notes

### Platform Support
- **Windows**: Fully tested and supported
- **Linux**: Not fully tested, may require modifications to scripts and paths
- **macOS**: Not tested

### Work in Progress
Flash Attention 2 integration is currently under development and not functional. This feature will be added in future updates to improve performance.

## Windows Installation (Recommended)

```bash
# Run the Windows installation script
setup.bat
```
This script will:
- Set up the required environment
- Install dependencies
- Configure the project for API usage

## Usage on Windows

### Starting the API Server
```bash
# Launch the WebSocket API server
tts-api.bat
```

The server will start on `ws://localhost:8000` by default.

### Testing the API
```bash
# Run the test script
test_api.bat
```

## Linux Support
If you want to run this on Linux, you'll need to:
1. Convert the `.bat` scripts to `.sh` scripts
2. Modify file paths to use forward slashes
3. Adjust environment setup commands
4. Test the WebSocket implementation in your environment

We welcome contributions to improve Linux support!

## WebSocket Protocol

### Messages Format

#### Client to Server:
```json
{
    "type": "tts_request",
    "data": {
        "text": "Text to synthesize",
        "reference_audio": "base64_encoded_audio",
        "config": {
            // Optional configuration parameters
        }
    }
}
```

#### Server to Client:
```json
{
    "type": "tts_response",
    "data": {
        "audio": "base64_encoded_audio",
        "status": "success",
        "message": "Optional status message"
    }
}
```

## Configuration

### Custom TOML Configuration
The system uses TOML files for configuration. Here's an example structure:

```toml
# Model selection
model = "F5TTS_Base"

# Audio input/output settings
ref_audio = "reference/reference.wav"
ref_text = "Reference text for the audio"
gen_text = "Text to be generated"

# Optional settings
remove_silence = false
output_dir = "tests"
output_file = "output.wav"
```

## Project Structure

```
F5-TTS/
├── api/                 # WebSocket API implementation
├── tests/              # Test files and output directory
├── reference/          # Reference audio files
├── setup.bat           # Windows installation script
├── test_api.bat        # Windows API testing script
├── tts-api.bat         # Windows server startup script
└── custom.toml         # Configuration file
```

### Directory Usage:
- `api/`: Contains the WebSocket server implementation
- `tests/`: Used for storing test outputs and test scripts
- `reference/`: Stores reference audio files for TTS generation

## Error Handling

The API includes robust error handling for common scenarios:
- Invalid audio format
- Missing configuration
- Server connection issues
- Processing errors

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Submitting a pull request

We especially welcome contributions for:
- Linux support improvements
- macOS compatibility
- Cross-platform testing

## License

This project is licensed under the MIT License.
