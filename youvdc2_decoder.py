import argparse
import cv2
import numpy as np
import os
import subprocess
import math
import librosa
from pydub import AudioSegment
from collections import Counter

# ---------- CONFIG ----------
BLOCK_SIZE = 50
MARGIN = 20
REPEAT_FRAMES = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
END_MARKER = "|||END|||"  # Special marker to detect message end

SAFE_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 .,!?_-'\"(){}\\/=\n" + "|"  # Added | for end marker
)

# Generate a safe 6×6×6 color cube (RGB) - well-spaced colors
LEVELS = [0, 51, 102, 153, 204, 255]
PALETTE = [(r, g, b) for r in LEVELS for g in LEVELS for b in LEVELS]
if len(PALETTE) < len(SAFE_ALPHABET):
    raise ValueError("Palette too small for alphabet!")

# ----------------------------

def char_to_color(c):
    """Map character to unique palette color."""
    if c not in SAFE_ALPHABET:
        raise ValueError(f"Unsupported character: {c}")
    idx = SAFE_ALPHABET.index(c)
    return PALETTE[idx]

def color_to_char(color):
    """Find nearest palette color and return matching char."""
    nearest_idx = None
    nearest_dist = float("inf")
    for i, pc in enumerate(PALETTE[:len(SAFE_ALPHABET)]):
        dist = sum((int(color[j]) - pc[j]) ** 2 for j in range(3))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_idx = i
    return SAFE_ALPHABET[nearest_idx]

def encode_block(frame, color):
    """Draw encoding block."""
    x, y = MARGIN, MARGIN
    frame[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = color
    return frame

def decode_block(frame):
    """Average block color."""
    x, y = MARGIN, MARGIN
    region = frame[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
    avg_color = np.mean(region.reshape(-1, 3), axis=0).astype(int)
    return tuple(avg_color)

def majority_vote(values):
    """Return most common value."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]

def extract_audio_features(audio_path, fps=30, max_duration=None):
    """Extract detailed audio features for visualization."""
    print("[+] Analyzing audio features...")
    
    # Load audio
    y, sr = librosa.load(audio_path, duration=max_duration)
    duration = len(y) / sr
    total_frames = int(duration * fps)
    
    # Extract features
    hop_length = len(y) // total_frames if total_frames > 0 else 512
    
    # Spectral features
    stft = librosa.stft(y, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Frequency bands
    freq_bins = magnitude.shape[0]
    bass_bins = freq_bins // 8      # Sub-bass and bass
    low_mid_bins = freq_bins // 4   # Low mids
    mid_bins = freq_bins // 2       # Mids
    high_mid_bins = 3 * freq_bins // 4  # High mids
    
    # Beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(tempo) if hasattr(tempo, '__len__') and len(tempo) > 0 else float(tempo)
    beat_frames = librosa.frames_to_samples(beats, hop_length=hop_length) * fps / sr
    
    features = []
    for frame_idx in range(min(total_frames, magnitude.shape[1])):
        # Energy in different frequency bands
        sub_bass = np.mean(magnitude[:bass_bins//2, frame_idx])
        bass = np.mean(magnitude[bass_bins//2:bass_bins, frame_idx])
        low_mid = np.mean(magnitude[bass_bins:low_mid_bins, frame_idx])
        mid = np.mean(magnitude[low_mid_bins:mid_bins, frame_idx])
        high_mid = np.mean(magnitude[mid_bins:high_mid_bins, frame_idx])
        treble = np.mean(magnitude[high_mid_bins:, frame_idx])
        
        # Overall energy
        total_energy = np.mean(magnitude[:, frame_idx])
        
        # Normalize
        if total_energy > 0:
            sub_bass /= total_energy
            bass /= total_energy
            low_mid /= total_energy
            mid /= total_energy
            high_mid /= total_energy
            treble /= total_energy
        
        # Beat detection
        is_beat = any(abs(frame_idx - beat_frame) < 2 for beat_frame in beat_frames)
        
        features.append({
            'sub_bass': min(1.0, sub_bass * 2),
            'bass': min(1.0, bass * 2),
            'low_mid': min(1.0, low_mid * 2),
            'mid': min(1.0, mid * 2),
            'high_mid': min(1.0, high_mid * 2),
            'treble': min(1.0, treble * 2),
            'total_energy': min(1.0, total_energy / np.max(magnitude)),
            'is_beat': is_beat,
            'frame_idx': frame_idx
        })
    
    print(f"[+] Extracted {len(features)} frames of audio features")
    return features, duration, tempo

def create_professional_frame(frame_idx, features, encoded_color, width=1280, height=720):
    """Create professional music visualizer frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get audio features
    sub_bass = features['sub_bass']
    bass = features['bass']
    low_mid = features['low_mid']
    mid = features['mid']
    high_mid = features['high_mid']
    treble = features['treble']
    total_energy = features['total_energy']
    is_beat = features['is_beat']
    
    center_x, center_y = width // 2, height // 2
    
    # Background gradient
    gradient_intensity = int(total_energy * 100)
    for y in range(height):
        color_val = max(0, gradient_intensity - (y * gradient_intensity // height))
        frame[y, :] = [color_val // 4, color_val // 6, color_val // 3]
    
    # Central circular visualizer
    base_radius = 120
    
    # Bass drop effect - expanding circles
    if bass > 0.7:
        for i in range(3):
            radius = int(base_radius + (bass * 200) + i * 30)
            thickness = max(1, int(bass * 15))
            alpha = 1.0 - (i * 0.3)
            color = (
                int(255 * alpha * bass),
                int(100 * alpha),
                int(200 * alpha * bass)
            )
            cv2.circle(frame, (center_x, center_y), radius, color, thickness)
    
    # Frequency spectrum circular bars
    num_bars = 64
    for i in range(num_bars):
        angle = (2 * np.pi * i / num_bars)
        
        # Different frequencies at different radii
        if i < num_bars // 4:
            intensity = sub_bass
            base_color = [255, 50, 50]  # Red for sub-bass
        elif i < num_bars // 2:
            intensity = bass
            base_color = [255, 150, 0]  # Orange for bass
        elif i < 3 * num_bars // 4:
            intensity = mid
            base_color = [100, 255, 100]  # Green for mids
        else:
            intensity = treble
            base_color = [100, 150, 255]  # Blue for treble
        
        # Bar length based on intensity
        bar_length = int(base_radius * 0.3 + intensity * base_radius * 0.8)
        
        # Positions
        start_radius = base_radius + 20
        end_radius = start_radius + bar_length
        
        start_x = center_x + int(start_radius * np.cos(angle))
        start_y = center_y + int(start_radius * np.sin(angle))
        end_x = center_x + int(end_radius * np.cos(angle))
        end_y = center_y + int(end_radius * np.sin(angle))
        
        # Color with intensity
        color = [int(c * intensity) for c in base_color]
        thickness = max(1, int(intensity * 8))
        
        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
    
    # Beat pulse effect
    if is_beat:
        pulse_radius = int(base_radius + total_energy * 100)
        cv2.circle(frame, (center_x, center_y), pulse_radius, (255, 255, 255), 3)
        
        # Beat flash
        overlay = np.ones_like(frame) * int(total_energy * 50)
        frame = cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)
    
    # Waveform visualization (bottom)
    wave_y = height - 80
    wave_points = []
    for i in range(width):
        # Create wave based on different frequencies
        wave_val = (
            sub_bass * np.sin(i * 0.02 + frame_idx * 0.1) * 30 +
            bass * np.sin(i * 0.04 + frame_idx * 0.15) * 20 +
            mid * np.sin(i * 0.08 + frame_idx * 0.2) * 15 +
            treble * np.sin(i * 0.16 + frame_idx * 0.25) * 10
        )
        wave_points.append([i, int(wave_y + wave_val)])
    
    # Draw waveform
    if len(wave_points) > 1:
        wave_points = np.array(wave_points, dtype=np.int32)
        cv2.polylines(frame, [wave_points], False, (0, 255, 255), 2)
    
    # Particle system for high frequencies
    if treble > 0.5:
        num_particles = int(treble * 50)
        for _ in range(num_particles):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height // 2)
            size = max(1, int(treble * 3))
            color = [
                int(255 * treble),
                int(200 * treble),
                255
            ]
            cv2.circle(frame, (x, y), size, color, -1)
    
    # Central logo/text area
    text_radius = int(base_radius * 0.6)
    cv2.circle(frame, (center_x, center_y), text_radius, (0, 0, 0), -1)
    cv2.circle(frame, (center_x, center_y), text_radius, (100, 100, 100), 2)
    
    # Dynamic text
    font_scale = 1.2
    text = "github.com/shiky8"
    text_size = cv2.getTextSize(text, FONT, font_scale, 2)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    
    # Text color based on audio
    text_color = (
        int(255 - bass * 100),
        int(255 - mid * 100),
        int(255 - treble * 100)
    )
    
    cv2.putText(frame, text, (text_x, text_y), FONT, font_scale, text_color, 2)
    
    # Encoding block (hidden in top-left)
    frame = encode_block(frame, encoded_color)
    
    # Add subtle border effect
    cv2.rectangle(frame, (0, 0), (width-1, height-1), (50, 50, 50), 2)
    
    return frame

def calculate_bitrate_for_target_size(duration, target_mb_min=3, target_mb_max=6):
    """Calculate video bitrate to achieve target file size."""
    # Convert MB to bits
    target_bits_min = target_mb_min * 8 * 1024 * 1024
    target_bits_max = target_mb_max * 8 * 1024 * 1024
    
    # Audio bitrate in bits per second (128k for good quality)
    audio_bitrate = 128 * 1024
    
    # Calculate available bits for video
    video_bits_min = target_bits_min - (audio_bitrate * duration)
    video_bits_max = target_bits_max - (audio_bitrate * duration)
    
    # Calculate video bitrates
    video_bitrate_min = max(300000, int(video_bits_min / duration))  # Minimum 300k
    video_bitrate_max = int(video_bits_max / duration)
    
    # Use target towards the higher end for better quality
    target_bitrate = int(video_bitrate_min + (video_bitrate_max - video_bitrate_min) * 0.7)
    
    return target_bitrate, video_bitrate_max

def create_video_with_message(audio_path, message, output_path, width=1280, height=720):
    """Create professional video with encoded message."""
    print(f"[+] Creating professional video with message: '{message}'")
    
    # Add end marker to message
    full_message = message + END_MARKER
    
    # Extract audio features
    audio_features, duration, tempo = extract_audio_features(audio_path)
    fps = 30
    frames_count = len(audio_features)
    
    print(f"[+] Duration: {duration:.1f}s, Tempo: {float(tempo):.1f} BPM, Frames: {frames_count}")
    
    # Calculate target bitrate for 3-6MB file size
    target_bitrate, max_bitrate = calculate_bitrate_for_target_size(duration)
    print(f"[+] Target video bitrate: {target_bitrate//1000}k (max: {max_bitrate//1000}k)")
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = 'temp_professional_video.mp4'
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    # Prepare message encoding
    colors = [char_to_color(c) for c in full_message]
    
    byte_index = 0
    repeat_counter = 0
    
    print("[+] Generating frames...")
    for i, features in enumerate(audio_features):
        if i % 100 == 0:
            print(f"Frame {i}/{frames_count}")
        
        # Create frame with current character's color
        frame = create_professional_frame(i, features, colors[byte_index], width, height)
        out.write(frame)
        
        # Advance to next character
        repeat_counter += 1
        if repeat_counter >= REPEAT_FRAMES:
            repeat_counter = 0
            byte_index = (byte_index + 1) % len(colors)
    
    out.release()
    
    # First pass - try with calculated bitrate
    print("[+] Mixing audio and video (optimizing size)...")
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", audio_path,
        "-c:v", "libx264", 
        "-b:v", f"{target_bitrate}",
        "-maxrate", f"{max_bitrate}",
        "-bufsize", f"{max_bitrate * 2}",
        "-c:a", "aac", "-b:a", "128k",
        "-preset", "medium", "-crf", "23",
        "-movflags", "+faststart",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(temp_video)
        
        # Check file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[+] Video size: {size_mb:.1f} MB")
        
        # If size is not in target range, adjust
        if size_mb > 6.5:
            print("[+] File too large, re-encoding with lower bitrate...")
            lower_bitrate = int(target_bitrate * 0.7)
            cmd[7] = f"{lower_bitrate}"  # Update bitrate
            cmd[9] = f"{lower_bitrate * 2}"  # Update maxrate
            subprocess.run(cmd, check=True, capture_output=True)
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[+] Adjusted video size: {size_mb:.1f} MB")
        elif size_mb < 2.5:
            print("[+] File too small, re-encoding with higher bitrate...")
            higher_bitrate = int(target_bitrate * 1.3)
            cmd[7] = f"{higher_bitrate}"  # Update bitrate
            cmd[9] = f"{higher_bitrate * 2}"  # Update maxrate
            cmd[13] = "20"  # Lower CRF for better quality
            subprocess.run(cmd, check=True, capture_output=True)
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[+] Adjusted video size: {size_mb:.1f} MB")
        
        if 3 <= size_mb <= 6:
            print(f"[+] Perfect! Video size: {size_mb:.1f} MB (within 3-6 MB target)")
        else:
            print(f"[+] Video created: {output_path} ({size_mb:.1f} MB)")
            
    except subprocess.CalledProcessError as e:
        print(f"[-] FFmpeg error: {e}")
        print("Using basic video without audio...")
        os.rename(temp_video, output_path)

def decode_message_from_video_auto(video_path):
    """Decode message automatically without knowing length."""
    print(f"[+] Auto-decoding message from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[-] Cannot open video file!")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[+] Video: {total_frames} frames at {fps} FPS")
    
    # Extract all colors first
    all_colors = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        avg_color = decode_block(frame)
        char_guess = color_to_char(avg_color)
        all_colors.append(char_guess)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    
    # Group by character position using REPEAT_FRAMES
    character_groups = []
    for i in range(0, len(all_colors), REPEAT_FRAMES):
        group = all_colors[i:i+REPEAT_FRAMES]
        if group:
            # Use majority voting for each character
            char = majority_vote(group)
            character_groups.append(char)
    
    # Reconstruct message
    message = "".join(character_groups)
    
    # Find end marker
    if END_MARKER in message:
        end_pos = message.find(END_MARKER)
        decoded_message = message[:end_pos]
        print(f"[+] Found end marker at position {end_pos}")
    else:
        # If no end marker, try to find a reasonable stopping point
        decoded_message = message.rstrip('|').rstrip()
        print("[*] No end marker found, using best guess")
    
    print(f"[+] Raw extracted: '{message[:50]}...' ({len(message)} chars)")
    return decoded_message

# -------- MAIN --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Music Visualizer with Steganography")
    parser.add_argument("audio", nargs="?", help="Audio file to use")
    parser.add_argument("-m", "--message", help="Message to encode")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("-df", "--decode_file", help="Video to decode (auto-detects length)")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    args = parser.parse_args()

    if args.decode_file:
        if not os.path.exists(args.decode_file):
            print(f"[-] Video file '{args.decode_file}' not found!")
        else:
            decoded = decode_message_from_video_auto(args.decode_file)
            if decoded:
                print(f"\n[+] DECODED MESSAGE: '{decoded}'\n")
            else:
                print("[-] Failed to decode message!")
    else:
        if not args.audio or not args.message or not args.output:
            print("[-] Need audio (-a), message (-m), and output (-o) for encoding")
            parser.print_help()
        elif not os.path.exists(args.audio):
            print(f"[-] Audio file '{args.audio}' not found!")
        else:
            create_video_with_message(args.audio, args.message, args.output, args.width, args.height)

# USAGE EXAMPLES:
# Encode: python3 pro_visualizer.py audio.mp3 -m "secret message" -o "output.mp4"
# Encoding takes 2 to 4 mins
# Decode: python3 pro_visualizer.py -df "output.mp4"
# Decoding takes 1 to 2 mins