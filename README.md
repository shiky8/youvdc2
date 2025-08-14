# YouTube videos Command & Control (youvdc2)

A sophisticated steganography-based command and control system that hides encrypted messages and commands inside professional-looking music visualizer videos on YouTube.

##  IMPORTANT DISCLAIMER

This tool is for educational and authorized security research purposes only. Users are responsible for complying with all applicable laws and regulations. Unauthorized use for malicious purposes is strictly prohibited.

##  Features

- **Advanced Steganography**: Encodes messages using color-mapped characters hidden in music visualizer videos
- **Professional Cover**: Creates legitimate-looking music visualizer content to avoid detection
- **YouTube Integration**: Full YouTube API integration for automated video upload/management
- **Command Execution**: Remote command execution through YouTube comments
- **Audio-Visual Synchronization**: Professional beat-synchronized visualizations with multiple frequency bands
- **Auto-Sizing**: Intelligent video compression to maintain 3-6MB file sizes for optimal upload
- **Robust Decoding**: Error-resistant decoding with majority voting and end marker detection

##  Installation

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# System dependencies (Ubuntu/Debian)
sudo apt-get install ffmpeg

# System dependencies (MacOS)
brew install ffmpeg
```

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shiky8/youvdc2.git
   cd youvdc2
   ```

2. **YouTube API Setup**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable YouTube Data API v3
   - Create OAuth 2.0 credentials
   - Download `client_secrets.json` and place in project directory

3. **Generate Authentication Token**:
   ```bash
   python3 youvdc2_get_token.py
   ```
   This will generate a `token.json` file with your YouTube API credentials.

4. **Configure Main Script**:
   - Edit `youvdc2.py` and update the `token_json_string` variable with the contents of your generated `token.json` file
   - Replace the hardcoded token data with your actual credentials

##  File Structure

- `youvdc2.py` - Main C2 server script (handles encoding and YouTube operations)
- `youvdc2_decoder.py` - Standalone video decoder only
- `youvdc2_get_token.py` - YouTube OAuth token generator
- `requirements.txt` - Python dependencies
- `wawa.mp3` - Audio file for video generation (you need to provide this)
- `client_secrets.json` - YouTube API credentials (you need to create this)
- `token.json` - Generated authentication token (created by get_token script)

##  Usage

### Server Mode (C2 Server)

1. **Prepare audio file**: Place your audio file as `wawa.mp3` in the project directory

2. **Run the server**:
   ```bash
   python3 youvdc2.py
   ```

3. **Operation**:
   - Server creates an initial video with "hi shiky8" message
   - Uploads to YouTube and waits for commands in comments
   - Executes commands from user `@mohamedshahat1817`
   - Returns command output via new encoded videos
   - Use comment "stopme" to terminate the server

### Standalone Decoding

**Decode a video**:
```bash
python3 youvdc2_decoder.py -df encoded_video.mp4
```

*Note: The decoder script is for decoding only. All encoding is handled by the main `youvdc2.py` script.*

### Advanced Options

```bash
# Decode videos with automatic message length detection
python3 youvdc2_decoder.py -df video.mp4
```

## Proof of Concept (PoC)
https://github.com/user-attachments/assets/e13ef990-569b-4045-b3ec-8e4dc3f0cf04

##  Technical Details

### Steganography Method

- **Color Mapping**: Uses a 6×6×6 RGB color cube (216 colors) for character encoding
- **Character Set**: Supports letters, numbers, punctuation, and special characters
- **Frame Repetition**: Each character is repeated across 5 frames for error resistance
- **End Marker**: Uses `|||END|||` to detect message boundaries
- **Block Location**: Hidden encoding block at position (20,20) with 50×50 pixel size

### Audio Analysis

- **Frequency Bands**: Sub-bass, bass, low-mid, mid, high-mid, treble
- **Beat Detection**: Real-time tempo and beat tracking
- **Visual Effects**: Beat-synchronized circles, frequency bars, particle systems
- **Waveform**: Dynamic bottom waveform based on frequency content

### Video Optimization

- **Target Size**: 3-6MB for optimal YouTube upload
- **Adaptive Bitrate**: Automatically adjusts video bitrate based on duration
- **Multiple Passes**: Re-encodes if file size is outside target range
- **Fast Start**: Optimized for streaming with movflags

##  Visualization Features

- Circular frequency spectrum analyzer (64 bars)
- Beat-synchronized expanding circles for bass drops
- Dynamic background gradients
- Particle effects for high frequencies  
- Multi-layered waveform visualization
- Professional branding area in center
- Audio-reactive color schemes

##  Security Considerations

- Uses legitimate-looking music visualizer as cover
- Messages are hidden in imperceptible color variations
- No obvious visual artifacts or suspicious patterns
- Automatic cleanup of temporary files
- Error-resistant encoding with majority voting

##  Configuration

Key parameters in the scripts:

```python
BLOCK_SIZE = 50          # Size of encoding block
MARGIN = 20              # Position from top-left corner  
REPEAT_FRAMES = 5        # Frames per character for redundancy
END_MARKER = "|||END|||" # Message termination marker
```

##  Troubleshooting

**FFmpeg not found**:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# MacOS  
brew install ffmpeg
```

**YouTube API quota exceeded**:
- Wait for quota reset (usually daily)
- Consider using multiple API keys
- Reduce upload frequency

**Audio analysis fails**:
```bash
pip install --upgrade librosa soundfile
```

**Video encoding errors**:
- Check audio file format (MP3 recommended)
- Ensure sufficient disk space
- Verify FFmpeg installation

##  Performance

- **Encoding Speed**: 2-4 minutes for typical 3-minute songs
- **Decoding Speed**: 1-2 minutes per video
- **File Size**: Automatically optimized to 3-6MB range
- **Quality**: 720p/1080p support with professional appearance

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

##  License

This project is for educational purposes only. Users are responsible for compliance with applicable laws and YouTube's Terms of Service.

##  Recent Updates

- Professional music visualizer with multiple visual effects
- Automatic file size optimization for YouTube uploads
- Enhanced audio analysis with beat detection
- Robust error correction with majority voting
- Improved visual aesthetics and cover legitimacy

---

**Remember**: This tool should only be used for authorized security research and educational purposes. Always comply with applicable laws and platform terms of service.
