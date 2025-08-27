"""
Voice Playback with Gemini CLI

This script uses the local Gemini CLI to play audio files from the Voice/ directory
according to the data in voiceData.csv. It provides eight modes of operation:

1. Direct mode: Gemini selects the audio file directly
2. Text matching mode: Match Gemini's response with voice data text
3. Manual selection mode: Select an audio file by number
4. List available models: Display all available Gemini models (for troubleshooting)
5. Speech recognition mode: Use microphone input instead of typing
6. Text-only mode: Interact with Gemini without audio playback
7. RAG mode: Use Gemini for conversation and local RAG for audio selection (token-efficient)
8. Speech RAG mode: Use speech input with Gemini for conversation and local RAG for audio (token-efficient)

Requirements:
- Python 3.7+
- pygame library (for audio playback)
- speech_recognition library (for speech recognition)
- python-dotenv library (for loading environment variables)
- Gemini CLI installed and configured on your system
- Access to one of these Gemini models through the CLI: 'gemini-2.5-pro', 'gemini-2.5-flash', or 'gemini-pro-vision'

Setup:
1. Install the Gemini CLI on your system
   - Follow the official Gemini CLI installation instructions
   - Make sure you've authenticated with the CLI before running this script
2. Configure the path to the Gemini CLI in the .env file:
   - Add GEMINI_CLI_PATH=<path_to_gemini_cli> to your .env file
   - Example: GEMINI_CLI_PATH=C:/Users/username/AppData/Roaming/npm/gemini.cmd
3. (Optional) Configure a custom system prompt:
   - Create a text file with your system prompt instructions
   - Add SYSTEM_PROMPT_PATH=<path_to_system_prompt_file> to your .env file
   - Example: SYSTEM_PROMPT_PATH=system_prompt.txt
   - The system prompt file should contain instructions for Gemini on how to select audio files
4. Install required Python packages: pip install pygame SpeechRecognition pyaudio python-dotenv

Usage:
- Run the script: python play_voice_with_gemini.py
- Select a mode (1, 2, 3, 4, 5, 6, 7, or 8) once at the beginning
- Continue the conversation with Gemini in a continuous chat session
- Type or say 'quit' to exit or 'change mode' to select a different mode

Performance Optimizations:
- The script now maintains conversation history for more contextual responses
- Conversation history is formatted in a clear, structured way for better context awareness
- History is automatically limited to prevent command line length issues
- Model instances are created once and reused for faster response times
- Continuous chat mode eliminates the need to select a mode for each interaction
- RAG modes (Modes 7 and 8) significantly reduce token usage by using Gemini only for conversation
  and handling audio selection locally with advanced text matching algorithms
- Speech RAG mode (Mode 8) combines the token efficiency of RAG with speech recognition for a
  fully voice-interactive experience that minimizes API usage

Quota Handling:
- The script includes automatic handling for API quota limits:
  - Retries with exponential backoff when quota limits are hit
  - Falls back to alternative models if the primary model's quota is exhausted
  - Provides clear error messages about quota limitations
- Free tier API keys have limited quota. If you're frequently hitting limits, consider:
  - Spacing out your requests
  - Upgrading to a paid API tier for higher quotas
  - Creating a new API key if your current one has exhausted its quota
"""

import os
import csv
import time
import pygame
import speech_recognition as sr
import subprocess
import datetime
import json
import tempfile
import re
import math
import base64
from pathlib import Path
from dotenv import load_dotenv
from zyphra import ZyphraClient

# Load environment variables from .env file
load_dotenv()

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Note: We're using the local Gemini CLI which uses your system's authentication
# No API key configuration is needed as the CLI handles authentication

def run_gemini_command(command_args):
    """
    Run a Gemini CLI command and return its output.

    Uses the GEMINI_CLI_PATH environment variable from the .env file to locate the Gemini CLI executable.
    If the environment variable is not set, falls back to a default path.

    Args:
        command_args: List of command arguments to pass to the Gemini CLI

    Returns:
        The standard output from the Gemini CLI command or an error message
    """
    try:
        # Get the Gemini CLI path from environment variable or use default
        gemini_cli_path = os.getenv('GEMINI_CLI_PATH', "C:\\Users\\seiri\\AppData\\Roaming\\npm\\gemini.cmd")

        # Filter out any duplicate or undashed parameters
        # This prevents both dashed and undashed versions of the same parameter from being passed
        filtered_args = []
        skip_next = False
        for i, arg in enumerate(command_args):
            if skip_next:
                skip_next = False
                continue

            # Skip undashed parameter names that have dashed equivalents
            if i < len(command_args) - 1 and arg in ["temperature", "maxOutputTokens", "topK", "topP"]:
                skip_next = True
                continue

            # Skip undashed parameters that might be passed without values
            if arg in ["temperature", "maxOutputTokens", "topK", "topP"]:
                continue

            # Skip dashed parameters that aren't supported by the CLI
            if arg in ["--temperature", "--max-output-tokens", "--top-k", "--top-p"]:
                skip_next = True
                continue

            filtered_args.append(arg)

        # Create the Gemini CLI command as a list
        command = [gemini_cli_path] + filtered_args

        # Run the command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8"
        )

        # Return the standard output
        return result.stdout

    except FileNotFoundError:
        return "Error: Gemini CLI not found. Make sure Gemini CLI is installed and the GEMINI_CLI_PATH in your .env file is correct."
    except subprocess.CalledProcessError as e:
        # Return command execution error details
        return f"Error occurred:\n{e.stderr}"

# Path to the voice data CSV and voice directory
VOICE_DATA_CSV = "voiceData.csv"
VOICE_DIR = "Voice"
CLONE_VOICE_DIR = os.path.join(VOICE_DIR, "CloneVoice")

# Create necessary directories if they don't exist
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(CLONE_VOICE_DIR, exist_ok=True)

def load_system_prompt():
    """
    Load the system prompt from the text file specified in the .env file.

    Returns:
        The system prompt as a string, or None if the file could not be read
    """
    try:
        # Get the system prompt path from environment variable or use default
        system_prompt_path = os.getenv('SYSTEM_PROMPT_PATH', 'system_prompt.txt')

        # Try different encodings since Japanese text might be encoded differently
        encodings = ['utf-8', 'shift-jis', 'euc-jp', 'iso-2022-jp']

        for encoding in encodings:
            try:
                with open(system_prompt_path, 'r', encoding=encoding) as file:
                    system_prompt = file.read()
                print(f"Successfully loaded system prompt with encoding: {encoding}")
                return system_prompt
            except UnicodeDecodeError:
                continue  # Try the next encoding
            except Exception as e:
                print(f"Error with encoding {encoding}: {e}")
                continue

        print(f"Could not read system prompt file with any encoding. Using default system prompt.")
        return None
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        return None

def load_voice_data():
    """Load the voice data from the CSV file."""
    voice_data = {}
    try:
        # Try different encodings since Japanese text might be encoded differently
        # Added cp932 which is commonly used for Japanese text on Windows
        encodings = ['utf-8', 'shift-jis', 'euc-jp', 'iso-2022-jp', 'cp932']

        for encoding in encodings:
            try:
                with open(VOICE_DATA_CSV, 'r', encoding=encoding) as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header row
                    for row in reader:
                        if len(row) >= 2:
                            filename, text = row[0], row[1]
                            # Skip empty filenames or rows with just a filename
                            if filename and filename.strip():
                                # Check if the text is actually readable
                                if text and len(text.strip()) > 0:
                                    voice_data[filename] = text
                print(f"エンコーディング {encoding} で音声データを正常に読み込みました")
                # Display a sample of the loaded data to verify it's readable
                sample_items = list(voice_data.items())[:3]
                for filename, text in sample_items:
                    print(f"サンプル: {filename} -> {text[:20]}...")
                break  # If we get here, we've successfully read the file
            except UnicodeDecodeError:
                continue  # Try the next encoding
            except Exception as e:
                print(f"エンコーディング {encoding} でエラー: {e}")
                continue
    except Exception as e:
        print(f"音声データの読み込みエラー: {e}")

    return voice_data

def create_voice_data_json(voice_data):
    """
    Create a JSON file with the voice data.

    Args:
        voice_data: Dictionary mapping filenames to their text content

    Returns:
        The path to the created JSON file
    """
    try:
        # Create a temporary directory that will persist until the program exits
        temp_dir = tempfile.mkdtemp()
        json_path = os.path.join(temp_dir, "voice_data.json")

        # Write the voice data to the JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(voice_data, f, ensure_ascii=False, indent=2)

        print(f"Created voice data JSON file at: {json_path}")
        return json_path
    except Exception as e:
        print(f"Error creating voice data JSON file: {e}")
        return None

def play_audio(filename):
    """Play the specified audio file."""
    try:
        # Check if the filename is an absolute path or just a filename
        if os.path.isabs(filename):
            file_path = filename
        else:
            file_path = os.path.join(VOICE_DIR, filename)

        # Check if the file exists
        if os.path.exists(file_path):
            print(f"Playing: {os.path.basename(file_path)}")
            try:
                sound = pygame.mixer.Sound(file_path)
                sound.play()
                # Wait for the audio to finish playing
                pygame.time.wait(int(sound.get_length() * 1000))
            except Exception as sound_error:
                print(f"Error playing sound: {sound_error}")
                # Try alternative method if pygame fails
                try:
                    import winsound
                    print("Trying to play with winsound...")
                    winsound.PlaySound(file_path, winsound.SND_FILENAME)
                except Exception as winsound_error:
                    print(f"Error playing with winsound: {winsound_error}")
        else:
            # Try to find the file in the current working directory
            cwd_path = os.path.join(os.getcwd(), os.path.basename(filename))
            if os.path.exists(cwd_path):
                print(f"Found audio file in current directory: {cwd_path}")
                return play_audio(cwd_path)  # Recursive call with absolute path
            else:
                print(f"Audio file not found: {file_path}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Checking if file exists in Voice directory...")
                voice_files = os.listdir(VOICE_DIR)
                print(f"Files in Voice directory: {voice_files[:10]}...")
    except Exception as e:
        print(f"Error playing audio: {e}")

def preprocess_text_for_zonos(text):
    """
    Preprocess text before sending to Zonos API.

    This function:
    1. Removes or replaces special characters that might cause issues
    2. Limits text length to a reasonable size
    3. Formats the text for better TTS results

    Args:
        text: The text to preprocess

    Returns:
        Preprocessed text
    """
    if not text:
        return text

    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)

    # Replace multiple spaces with a single one
    text = re.sub(r' +', ' ', text)

    # Remove special characters that might cause issues (keep Japanese characters)
    text = re.sub(r'[^\w\s\.,;:!?。、；：！？「」『』（）\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', text)

    # Limit text length (3000 characters should be safe)
    if len(text) > 3000:
        print(f"テキストが長すぎるため切り詰めます: {len(text)} → 3000文字")
        text = text[:2997] + "..."

    return text.strip()

def detect_language_from_system_prompt(system_prompt=None):
    """
    Detect the language from the system prompt.

    Args:
        system_prompt: The system prompt text

    Returns:
        A tuple of (language_code, language_name)
        language_code: ISO language code (e.g., 'ja', 'en')
        language_name: Human-readable language name (e.g., 'Japanese', 'English')
    """
    if not system_prompt:
        # Try to load the system prompt if not provided
        system_prompt = load_system_prompt()

    if not system_prompt:
        # Default to Japanese if no system prompt is available
        return 'ja', 'Japanese'

    # Count characters in different scripts
    japanese_chars = sum(1 for c in system_prompt if ord(c) >= 0x3000 and ord(c) <= 0x9FFF)
    english_chars = sum(1 for c in system_prompt if ord(c) >= 0x0020 and ord(c) <= 0x007F)

    # Calculate percentages
    total_chars = len(system_prompt)
    japanese_percentage = japanese_chars / total_chars * 100 if total_chars > 0 else 0
    english_percentage = english_chars / total_chars * 100 if total_chars > 0 else 0

    print(f"Language detection: Japanese {japanese_percentage:.1f}%, English {english_percentage:.1f}%")

    # Determine the dominant language
    if japanese_percentage > 15:  # Even a small percentage of Japanese characters indicates Japanese content
        return 'ja', 'Japanese'
    elif english_percentage > 50:
        return 'en', 'English'
    else:
        # Default to Japanese for other cases
        return 'ja', 'Japanese'

def generate_zonos_voice_data(text, voice_id="default", use_clone_voice=False, clone_voice_file=None, max_retries=3, language_code=None, system_prompt=None):
    """
    Generate voice data using Zonos API via the Zyphra client library.
    This function handles only the API call to generate the audio data.

    Args:
        text: Text to convert to speech
        voice_id: Zonos voice ID to use (default will be used if not specified)
        use_clone_voice: Whether to use a clone voice
        clone_voice_file: The filename of the clone voice to use
        max_retries: Maximum number of retry attempts for transient errors
        language_code: ISO language code (e.g., 'ja', 'en') to use for TTS
        system_prompt: The system prompt to use for language detection if language_code is not provided

    Returns:
        Audio data as bytes or None if generation failed
    """
    try:
        # Log the original text from Gemini
        print(f"\n===== Gemini Response Text (Before Preprocessing) =====")
        print(f"Length: {len(text)} characters")
        print(f"Text: {text[:500]}..." if len(text) > 500 else f"Text: {text}")

        # Preprocess the text before sending to Zonos
        processed_text = preprocess_text_for_zonos(text)

        # Log the preprocessed text
        if processed_text != text:
            print(f"\n===== Preprocessed Text for Zonos =====")
            print(f"Length: {len(processed_text)} characters")
            print(f"Text: {processed_text[:500]}..." if len(processed_text) > 500 else f"Text: {processed_text}")

        # Log to file for later analysis
        log_gemini_response_for_tts(text, processed_text)

        # Use the preprocessed text for the rest of the function
        text = processed_text

        # Validate the text before sending to Zonos
        is_valid, validation_message = validate_zonos_text(text)
        if not is_valid:
            print(f"テキスト検証エラー: {validation_message}")
            return None

        # Get API key from environment variables
        api_key = os.getenv('ZONOS_API')
        if not api_key:
            print("Zonos API key not found in environment variables")
            return None

        # Initialize the Zyphra client
        client = ZyphraClient(api_key=api_key)

        # Prepare parameters for the API call
        params = {
            "text": text,
            "speaking_rate": 15,
            "model": "zonos-v0.1-transformer"
        }

        # Determine the language to use
        detected_language_code = None

        # Use provided language_code if available
        if language_code:
            detected_language_code = language_code
            print(f"Using provided language code: {language_code}")
        # Otherwise detect from system prompt
        elif system_prompt:
            detected_language_code, language_name = detect_language_from_system_prompt(system_prompt)
            print(f"Detected language from system prompt: {language_name} ({detected_language_code})")
        # Otherwise detect from text content
        else:
            # Check for Japanese characters in the text
            has_japanese = any(ord(c) >= 0x3000 for c in text)
            detected_language_code = 'ja' if has_japanese else 'en'
            print(f"Detected language from text content: {'Japanese' if has_japanese else 'English'} ({detected_language_code})")

        # Apply language-specific settings
        if detected_language_code == 'ja':
            print("Using Japanese language settings with hybrid model")
            params["model"] = "zonos-v0.1-hybrid"
            params["language_iso_code"] = "ja"
            # Japanese performs better with higher speaking rates
            params["speaking_rate"] = 18
        elif detected_language_code == 'en':
            print("Using English language settings with transformer model")
            params["model"] = "zonos-v0.1-transformer"
            params["language_iso_code"] = "en"
            params["speaking_rate"] = 15
        else:
            # Default to hybrid model for other languages
            print(f"Using default settings for language: {detected_language_code}")
            params["model"] = "zonos-v0.1-hybrid"
            params["language_iso_code"] = detected_language_code
            params["speaking_rate"] = 15

        # Initialize variables for retry loop
        retry_count = 0
        success = False
        last_error = None
        audio_data = None

        # Retry loop for transient errors
        while retry_count < max_retries and not success:
            try:
                if retry_count > 0:
                    # Add exponential backoff delay for retries
                    delay = 2 ** retry_count  # 2, 4, 8 seconds
                    print(f"Retry attempt {retry_count}/{max_retries} after {delay} seconds...")
                    time.sleep(delay)

                # If using a clone voice, prepare the clone voice file
                if use_clone_voice and clone_voice_file:
                    clone_voice_path = os.path.join(CLONE_VOICE_DIR, clone_voice_file)
                    if os.path.exists(clone_voice_path):
                        # Read and encode the clone voice file
                        with open(clone_voice_path, "rb") as f:
                            clone_audio_data = f.read()
                            speaker_audio = base64.b64encode(clone_audio_data).decode('utf-8')

                        # Add to parameters
                        params["speaker_audio"] = speaker_audio

                        # For better results with cloned voices
                        params["vqscore"] = 0.7  # Controls voice quality vs. speaker similarity
                        params["speaker_noised"] = True  # Improves stability

                        print(f"Using clone voice: {clone_voice_file}")
                    else:
                        print(f"Clone voice file not found: {clone_voice_path}")

                # Print debug information
                print(f"Zyphra client parameters: {params}")

                # Make the API call using the client library
                audio_data = client.audio.speech.create(**params)
                success = True

            except Exception as e:
                retry_count += 1
                last_error = str(e)
                print(f"Error on attempt {retry_count}/{max_retries}: {e}")

                # Check if we should retry
                if retry_count >= max_retries:
                    raise Exception(f"Max retries ({max_retries}) exceeded. Last error: {last_error}")

        # Return the audio data
        if audio_data:
            print(f"Zyphra client response successful, audio data size: {len(audio_data)} bytes")
            return audio_data
        else:
            print("No audio data received from Zyphra client")
            return None

    except Exception as e:
        error_message = f"Error generating voice with Zyphra client: {e}"
        print(error_message)

        # Get detailed traceback for logging
        import traceback
        traceback_str = traceback.format_exc()

        # Log the error with traceback
        error_info = f"GENERAL ERROR:\n{error_message}\n\nTraceback:\n{traceback_str}"
        log_gemini_response_for_tts(gemini_response=text, processed_text=processed_text, error_info=error_info)

        return None

def save_zonos_voice_data(audio_data, text):
    """
    Save the generated audio data to a file and update the voice data CSV.

    Args:
        audio_data: The audio data as bytes
        text: The text content of the audio

    Returns:
        The filename of the saved audio file or None if saving failed
    """
    try:
        if not audio_data:
            print("No audio data to save")
            return None

        # Generate a filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"audio{timestamp}.webm"
        file_path = os.path.join(VOICE_DIR, filename)

        print(f"Saving audio data to: {file_path}")

        # Save audio file
        with open(file_path, "wb") as f:
            f.write(audio_data)

        # Update voice data CSV
        update_voice_data_csv(filename, text)

        # Schedule wav conversion in the background
        threading.Thread(target=convert_webm_to_wav, args=(file_path,)).start()

        return filename
    except Exception as e:
        print(f"Error saving audio data: {e}")
        return None

def convert_webm_to_wav(webm_path):
    """
    Convert a webm file to wav format using pydub.
    This function is designed to be run in a background thread.

    Args:
        webm_path: Path to the webm file to convert

    Returns:
        Path to the converted wav file or None if conversion failed
    """
    try:
        # Check if the file exists
        if not os.path.exists(webm_path):
            print(f"Webm file not found: {webm_path}")
            return None

        # Generate the output path
        wav_path = os.path.splitext(webm_path)[0] + ".wav"

        # Check if pydub is available
        try:
            from pydub import AudioSegment

            print(f"Converting {webm_path} to {wav_path} using pydub...")

            # Load the webm file
            audio = AudioSegment.from_file(webm_path, format="webm")

            # Export as wav
            audio.export(wav_path, format="wav")

            print(f"Conversion successful: {wav_path}")
            return wav_path
        except ImportError:
            # If pydub is not available, try using ffmpeg directly
            print("Pydub not available, trying ffmpeg directly...")

            # Check if ffmpeg is available
            try:
                # Construct the ffmpeg command
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", webm_path,
                    "-y",  # Overwrite output file if it exists
                    wav_path
                ]

                # Run the command
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                print(f"Conversion successful: {wav_path}")
                return wav_path
            except subprocess.CalledProcessError as e:
                print(f"Error running ffmpeg: {e}")
                print(f"ffmpeg stderr: {e.stderr}")
                return None
            except Exception as e:
                print(f"Error using ffmpeg: {e}")
                return None
    except Exception as e:
        print(f"Error converting webm to wav: {e}")
        return None

def batch_convert_webm_to_wav():
    """
    Batch convert all webm files in the Voice directory to wav format.
    This function can be called manually or scheduled to run periodically.
    """
    try:
        print("Starting batch conversion of webm files to wav...")

        # Get all webm files in the Voice directory
        webm_files = [f for f in os.listdir(VOICE_DIR) if f.endswith('.webm')]

        if not webm_files:
            print("No webm files found in Voice directory")
            return

        print(f"Found {len(webm_files)} webm files to convert")

        # Convert each file
        for webm_file in webm_files:
            webm_path = os.path.join(VOICE_DIR, webm_file)
            wav_path = os.path.splitext(webm_path)[0] + ".wav"

            # Skip if wav file already exists
            if os.path.exists(wav_path):
                print(f"Skipping {webm_file} - wav file already exists")
                continue

            # Convert the file
            convert_webm_to_wav(webm_path)

        print("Batch conversion completed")
    except Exception as e:
        print(f"Error in batch conversion: {e}")

def generate_zonos_voice(text, voice_id="default", use_clone_voice=False, clone_voice_file=None, max_retries=3, language_code=None, system_prompt=None):
    """
    Generate voice using Zonos API via the Zyphra client library.
    This is a wrapper function that combines generate_zonos_voice_data and save_zonos_voice_data.

    Args:
        text: Text to convert to speech
        voice_id: Zonos voice ID to use (default will be used if not specified)
        use_clone_voice: Whether to use a clone voice
        clone_voice_file: The filename of the clone voice to use
        max_retries: Maximum number of retry attempts for transient errors
        language_code: ISO language code (e.g., 'ja', 'en') to use for TTS
        system_prompt: The system prompt to use for language detection if language_code is not provided

    Returns:
        Path to the generated audio file or None if generation failed
    """
    # If system_prompt is not provided, try to load it
    if not system_prompt and not language_code:
        system_prompt = load_system_prompt()

    # Generate the audio data
    audio_data = generate_zonos_voice_data(
        text, 
        voice_id=voice_id, 
        use_clone_voice=use_clone_voice, 
        clone_voice_file=clone_voice_file, 
        max_retries=max_retries,
        language_code=language_code,
        system_prompt=system_prompt
    )

    # If we have audio data, save it to a file
    if audio_data:
        return save_zonos_voice_data(audio_data, text)
    else:
        return None

def test_zonos_connection():
    """
    Test the connection to the Zonos API and verify that the API key is valid using the Zyphra client library.

    Returns:
        bool: True if the connection is successful, False otherwise
        str: A message describing the result of the test
    """
    try:
        # Get API key from environment variables
        api_key = os.getenv('ZONOS_API')
        if not api_key:
            return False, "Zonos APIキーが環境変数に見つかりません"

        # Minimal parameters for test
        params = {
            "text": "テスト",
            "speaking_rate": 15,
            "model": "zonos-v0.1-transformer"
        }

        # Print test information
        print(f"Zyphra クライアント接続テスト")
        print(f"パラメータ: {params}")

        try:
            # Initialize the Zyphra client
            client = ZyphraClient(api_key=api_key)

            # Make a simple API call to test the connection
            # We don't need to save the result, just check if it works
            audio_data = client.audio.speech.create(**params)

            # If we get here, the connection was successful
            print(f"Zyphra クライアント接続テスト成功")
            print(f"オーディオデータサイズ: {len(audio_data)} バイト")

            return True, "Zonos APIへの接続に成功しました"

        except Exception as client_error:
            error_message = str(client_error)
            print(f"Zyphra クライアントエラー: {error_message}")

            # Check for common error types
            if "401" in error_message or "unauthorized" in error_message.lower():
                return False, "認証エラー: APIキーが無効です"
            elif "404" in error_message or "not found" in error_message.lower():
                return False, "エンドポイントが見つかりません: APIエンドポイントが正しいか確認してください"
            elif "429" in error_message or "too many requests" in error_message.lower():
                return False, "レート制限エラー: APIリクエストの頻度を下げてください"
            else:
                return False, f"Zyphra クライアントエラー: {error_message}"

    except Exception as e:
        return False, f"予期せぬエラー: {e}"

def validate_zonos_text(text):
    """
    Validate text to be sent to the Zonos API.

    Args:
        text: The text to validate

    Returns:
        bool: True if the text is valid, False otherwise
        str: A message describing the result of the validation
    """
    if not text:
        return False, "テキストが空です"

    if len(text) > 5000:  # Arbitrary limit, check API documentation for actual limit
        return False, f"テキストが長すぎます ({len(text)} 文字). 5000文字以下にしてください"

    # Check for special characters or control characters
    invalid_chars = [char for char in text if ord(char) < 32 and char not in ['\n', '\t', '\r']]
    if invalid_chars:
        return False, f"テキストに無効な制御文字が含まれています: {invalid_chars}"

    return True, "テキストは有効です"

def get_available_clone_voices():
    """
    Get a list of available clone voices from the CloneVoice directory.

    Returns:
        A list of dictionaries containing voice information
    """
    voices = []
    try:
        # Check if the directory exists
        if not os.path.exists(CLONE_VOICE_DIR):
            return voices

        # Get all files in the directory
        for filename in os.listdir(CLONE_VOICE_DIR):
            file_path = os.path.join(CLONE_VOICE_DIR, filename)
            # Only include files, not directories
            if os.path.isfile(file_path):
                # Get the file extension
                _, ext = os.path.splitext(filename)
                # Only include audio files
                if ext.lower() in ['.wav', '.mp3', '.ogg']:
                    # Create a voice entry
                    voice_name = os.path.splitext(filename)[0]
                    voices.append({
                        'id': filename,
                        'name': voice_name
                    })

        return voices
    except Exception as e:
        print(f"Error getting available clone voices: {e}")
        return []

def get_next_audio_number():
    """
    voiceData.csvから最大の連番を見つけて、次の番号を返す

    Returns:
        int: 次に使用する連番
    """
    try:
        max_number = 0

        # CSVファイルが存在するか確認
        if os.path.exists(VOICE_DATA_CSV):
            # Try different encodings for reading the file
            encodings = ['utf-8', 'shift-jis', 'euc-jp', 'iso-2022-jp', 'cp932']

            for encoding in encodings:
                try:
                    with open(VOICE_DATA_CSV, 'r', encoding=encoding) as file:
                        reader = csv.reader(file)
                        next(reader, None)  # ヘッダーをスキップ

                        for row in reader:
                            if not row:
                                continue

                            filename = row[0]
                            # "audio数字.拡張子" の形式からファイル番号を抽出
                            match = re.match(r'audio(\d+)\.\w+', filename)
                            if match:
                                number = int(match.group(1))
                                max_number = max(max_number, number)

                    print(f"Successfully read CSV with encoding: {encoding}")
                    break  # If we get here, we've successfully read the file
                except UnicodeDecodeError:
                    continue  # Try the next encoding
                except Exception as e:
                    print(f"Error reading CSV with encoding {encoding}: {e}")
                    continue

        # 次の番号を返す
        return max_number + 1

    except Exception as e:
        print(f"連番取得エラー: {e}")
        # エラーが発生した場合は、タイムスタンプを使用
        return int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

def update_voice_data_csv(filename, text):
    """
    Update the voiceData.csv file with new voice data.

    Args:
        filename: Name of the audio file
        text: Text content of the audio
    """
    try:
        # Read existing data
        existing_data = []
        file_encoding = 'utf-8'  # Default encoding

        # Try different encodings for reading the file
        encodings = ['utf-8', 'shift-jis', 'euc-jp', 'iso-2022-jp', 'cp932']
        file_exists = os.path.exists(VOICE_DATA_CSV)

        if file_exists:
            for encoding in encodings:
                try:
                    with open(VOICE_DATA_CSV, 'r', encoding=encoding) as file:
                        reader = csv.reader(file)
                        existing_data = list(reader)
                    print(f"Successfully read CSV with encoding: {encoding}")
                    file_encoding = encoding  # Remember the successful encoding
                    break  # If we get here, we've successfully read the file
                except UnicodeDecodeError:
                    continue  # Try the next encoding
                except Exception as e:
                    print(f"Error reading CSV with encoding {encoding}: {e}")
                    continue

        # If file doesn't exist or couldn't be read with any encoding
        if not file_exists or not existing_data:
            # Create file with header if it doesn't exist
            existing_data = [["filename", "text"]]
            print("Creating new CSV file with header")

        # Add new data
        existing_data.append([filename, text])

        # Write updated data using the same encoding that was successful for reading
        with open(VOICE_DATA_CSV, 'w', encoding=file_encoding, newline='') as file:
            writer = csv.writer(file)
            writer.writerows(existing_data)

        print(f"Updated voice data CSV with new entry: {filename} using encoding: {file_encoding}")

    except Exception as e:
        print(f"Error updating voice data CSV: {e}")

def list_available_models():
    """
    List all available models from the Gemini CLI.
    This is useful for troubleshooting when models are not found.

    Returns:
        A list of available model names
    """
    try:
        # Run the CLI command to list available models
        # The exact command may vary depending on the Gemini CLI implementation
        # This is a common pattern for CLI tools
        response = run_gemini_command(["--list", "models"])

        if response.startswith("Error:"):
            print(f"Error listing models: {response}")

            # Try alternative command format if the first one fails
            response = run_gemini_command(["--models", "list"])

            if response.startswith("Error:"):
                print(f"Error listing models with alternative command: {response}")
                return []

        # Parse the response to extract model names
        # This parsing logic may need to be adjusted based on the actual output format
        available_models = []
        for line in response.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("Error:"):
                # Assuming each line contains a model name, possibly with additional info
                # Extract just the model name (this may need adjustment)
                model_name = line.split()[0] if line.split() else line
                available_models.append(model_name)
                print(f"Available model: {model_name}")

        # If we couldn't parse any models but got a response, print the raw response
        if not available_models and response:
            print("Received response but couldn't parse model names. Raw output:")
            print(response)

            # Initialize models directly as a fallback
            print("Falling back to direct model checking...")
            initialize_models()
            return CLI_AVAILABLE_MODELS

        return available_models
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

# Variables for CLI-based approach
CLI_AVAILABLE_MODELS = []

def initialize_models():
    """
    Check which Gemini models are available through the CLI.
    This function tests the CLI connection and identifies available models.
    """
    global CLI_AVAILABLE_MODELS
    models_to_check = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-pro-vision']

    print("Checking available Gemini CLI models...")

    # Test the CLI connection first
    test_response = run_gemini_command(["--version"])
    if test_response.startswith("Error:"):
        print("Warning: Gemini CLI may not be properly installed or configured.")
        print(test_response)
        return

    # Check each model by running a simple test command
    for model_name in models_to_check:
        try:
            print(f"Checking model: {model_name}")
            # Simple test command to check if the model is available
            test_cmd = ["--model", model_name, "-p", "Hello"]
            response = run_gemini_command(test_cmd)

            if not response.startswith("Error:"):
                CLI_AVAILABLE_MODELS.append(model_name)
                print(f"Model {model_name} is available")
            else:
                print(f"Model {model_name} is not available: {response}")
        except Exception as e:
            print(f"Error checking model {model_name}: {e}")

def estimate_tokens(text):
    """
    Enhanced token estimation with improved accuracy for Japanese and English text.
    Uses more sophisticated heuristics based on character types, patterns, and language features.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0

    # Cache token estimations to avoid recalculating
    if not hasattr(estimate_tokens, 'cache'):
        estimate_tokens.cache = {}

    # Use a hash of the first 100 chars as cache key to keep size reasonable
    cache_key = hash(text[:100] + str(len(text)))

    if cache_key in estimate_tokens.cache:
        return estimate_tokens.cache[cache_key]

    # Count characters by type with more granularity
    kanji_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    hiragana_chars = sum(1 for c in text if '\u3040' <= c <= '\u309f')
    katakana_chars = sum(1 for c in text if '\u30a0' <= c <= '\u30ff')
    jp_punctuation = sum(1 for c in text if c in '、。！？…「」『』（）')
    digits = sum(1 for c in text if c.isdigit())
    spaces = sum(1 for c in text if c.isspace())
    en_chars = len(text) - kanji_chars - hiragana_chars - katakana_chars - jp_punctuation - digits - spaces

    # More accurate token estimation by character type
    # These values are based on empirical observations of tokenization patterns
    kanji_tokens = kanji_chars * 1.8       # Kanji characters use more tokens
    hiragana_tokens = hiragana_chars * 0.5  # Hiragana often groups into tokens
    katakana_tokens = katakana_chars * 1.2  # Katakana uses more tokens than hiragana
    jp_punct_tokens = jp_punctuation * 1.0  # Japanese punctuation is typically 1 token each
    digit_tokens = digits * 0.5             # Digits often group into tokens
    space_tokens = spaces * 0.1             # Spaces use minimal tokens
    en_tokens = en_chars * 0.25             # English uses ~4 characters per token on average

    # Adjust for common patterns that affect tokenization

    # 1. Repeated characters often use fewer tokens
    repeated_char_count = 0
    for i in range(1, len(text)):
        if text[i] == text[i-1]:
            repeated_char_count += 1

    # 2. Common Japanese particles use fewer tokens
    common_particles = ['は', 'を', 'に', 'の', 'と', 'が', 'で', 'も']
    particle_count = sum(text.count(p) for p in common_particles)

    # 3. Long numbers use fewer tokens per digit
    import re
    number_groups = re.findall(r'\d{4,}', text)
    long_digit_count = sum(len(group) for group in number_groups)

    # Apply adjustments
    token_adjustment = (
        - (repeated_char_count * 0.2)      # Reduce for repeated characters
        - (particle_count * 0.2)           # Reduce for common particles
        - (long_digit_count * 0.2)         # Reduce for long number sequences
    )

    # Calculate total tokens with adjustments
    total_tokens = (
        kanji_tokens +
        hiragana_tokens +
        katakana_tokens +
        jp_punct_tokens +
        digit_tokens +
        space_tokens +
        en_tokens +
        token_adjustment
    )

    # Ensure minimum token count and round to integer
    result = max(1, int(total_tokens))

    # Cache the result
    estimate_tokens.cache[cache_key] = result

    # Limit cache size
    if len(estimate_tokens.cache) > 1000:
        # Remove a random key
        estimate_tokens.cache.pop(next(iter(estimate_tokens.cache)))

    return result

def log_token_usage(prompt, response, model_name):
    """
    Log detailed token usage statistics for monitoring and optimization.

    Args:
        prompt: The prompt sent to the model
        response: The response received from the model
        model_name: The name of the model used

    Returns:
        A dictionary with token usage statistics
    """
    # Define quota limits for different models (tokens per day)
    # These are approximate values and may need adjustment based on actual limits
    QUOTA_LIMITS = {
        'gemini-2.5-pro': 60000,  # Example limit for Pro model
        'gemini-2.5-flash': 120000,  # Example limit for Flash model
        'gemini-pro-vision': 60000,  # Example limit for Vision model
        'default': 60000  # Default limit for unknown models
    }

    # Initialize token usage tracking if it doesn't exist
    if not hasattr(log_token_usage, 'session_stats'):
        log_token_usage.session_stats = {
            'total_prompt_tokens': 0,
            'total_response_tokens': 0,
            'total_tokens': 0,
            'model_usage': {},
            'request_count': 0,
            'avg_tokens_per_request': 0,
            'start_time': datetime.datetime.now(),
            'hourly_usage': {},
            'daily_usage': {},
            'token_savings': 0,  # Track estimated token savings from optimizations
            'quota_limits': QUOTA_LIMITS  # Store quota limits
        }

    # Estimate tokens for prompt and response
    prompt_tokens = estimate_tokens(prompt)
    response_tokens = estimate_tokens(response)
    total_tokens = prompt_tokens + response_tokens

    # Update session statistics
    log_token_usage.session_stats['total_prompt_tokens'] += prompt_tokens
    log_token_usage.session_stats['total_response_tokens'] += response_tokens
    log_token_usage.session_stats['total_tokens'] += total_tokens
    log_token_usage.session_stats['request_count'] += 1

    # Update model-specific usage
    if model_name not in log_token_usage.session_stats['model_usage']:
        log_token_usage.session_stats['model_usage'][model_name] = 0
    log_token_usage.session_stats['model_usage'][model_name] += total_tokens

    # Update hourly usage
    current_hour = datetime.datetime.now().strftime('%Y-%m-%d %H:00')
    if current_hour not in log_token_usage.session_stats['hourly_usage']:
        log_token_usage.session_stats['hourly_usage'][current_hour] = 0
    log_token_usage.session_stats['hourly_usage'][current_hour] += total_tokens

    # Update daily usage
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if current_date not in log_token_usage.session_stats['daily_usage']:
        log_token_usage.session_stats['daily_usage'][current_date] = {}

    # Initialize model usage for the day if not exists
    if model_name not in log_token_usage.session_stats['daily_usage'][current_date]:
        log_token_usage.session_stats['daily_usage'][current_date][model_name] = 0

    # Update model usage for the day
    log_token_usage.session_stats['daily_usage'][current_date][model_name] += total_tokens

    # Calculate average tokens per request
    log_token_usage.session_stats['avg_tokens_per_request'] = (
        log_token_usage.session_stats['total_tokens'] / 
        log_token_usage.session_stats['request_count']
    )

    # Estimate token savings from optimizations
    # Assuming unoptimized prompts would be ~30% longer
    unoptimized_prompt_estimate = int(prompt_tokens * 1.3)
    log_token_usage.session_stats['token_savings'] += (unoptimized_prompt_estimate - prompt_tokens)

    # Create current request stats
    current_stats = {
        'prompt_tokens': prompt_tokens,
        'response_tokens': response_tokens,
        'total_tokens': total_tokens,
        'model': model_name,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Calculate quota usage
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    daily_model_usage = log_token_usage.session_stats['daily_usage'].get(current_date, {}).get(model_name, 0)

    # Get quota limit for the model
    quota_limit = log_token_usage.session_stats['quota_limits'].get(
        model_name, 
        log_token_usage.session_stats['quota_limits']['default']
    )

    # Calculate percentage of quota used
    quota_percentage = (daily_model_usage / quota_limit) * 100 if quota_limit > 0 else 0

    # Log detailed statistics
    print(f"\n===== Token Usage Statistics =====")
    print(f"Current request: {prompt_tokens} prompt + {response_tokens} response = {total_tokens} tokens")
    print(f"Model used: {model_name}")
    print(f"Session total: {log_token_usage.session_stats['total_tokens']} tokens")
    print(f"Estimated token savings: {log_token_usage.session_stats['token_savings']} tokens")
    print(f"Average per request: {log_token_usage.session_stats['avg_tokens_per_request']:.1f} tokens")
    print(f"Request count: {log_token_usage.session_stats['request_count']}")
    print(f"\n===== Quota Usage ({current_date}) =====")
    print(f"Daily usage for {model_name}: {daily_model_usage} tokens")
    print(f"Quota limit: {quota_limit} tokens")
    print(f"Quota used: {quota_percentage:.2f}%")
    print(f"Remaining: {quota_limit - daily_model_usage} tokens ({100 - quota_percentage:.2f}%)")

    # Update current stats with quota information
    current_stats.update({
        'daily_usage': daily_model_usage,
        'quota_limit': quota_limit,
        'quota_percentage': quota_percentage,
        'quota_remaining': quota_limit - daily_model_usage
    })

    return current_stats

def get_token_usage_stats():
    """
    Get statistics about token usage.

    Returns:
        A formatted string with token usage statistics
    """
    if not hasattr(get_gemini_response, 'token_usage'):
        return "No token usage data available."

    usage = get_gemini_response.token_usage
    stats = [
        "===== Token Usage Statistics =====",
        f"Total tokens used: {usage['total_tokens']}",
        f"  - Prompt tokens: {usage['total_prompt_tokens']}",
        f"  - Response tokens: {usage['total_response_tokens']}",
        "\nUsage by model:"
    ]

    for model, model_usage in usage['model_usage'].items():
        stats.append(f"  {model}: {model_usage} tokens")

    # Add quota usage information if available
    if hasattr(log_token_usage, 'session_stats') and 'daily_usage' in log_token_usage.session_stats:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        daily_usage = log_token_usage.session_stats['daily_usage'].get(current_date, {})
        quota_limits = log_token_usage.session_stats.get('quota_limits', {})

        if daily_usage:
            stats.append(f"\n===== Quota Usage ({current_date}) =====")
            for model, model_usage in daily_usage.items():
                quota_limit = quota_limits.get(model, quota_limits.get('default', 60000))
                quota_percentage = (model_usage / quota_limit) * 100 if quota_limit > 0 else 0
                remaining = quota_limit - model_usage
                remaining_percentage = 100 - quota_percentage

                stats.append(f"  {model}:")
                stats.append(f"    - Daily usage: {model_usage} tokens")
                stats.append(f"    - Quota limit: {quota_limit} tokens")
                stats.append(f"    - Quota used: {quota_percentage:.2f}%")
                stats.append(f"    - Remaining: {remaining} tokens ({remaining_percentage:.2f}%)")

    stats.append("\nNote: These are estimated values and may differ from actual token counts.")
    return "\n".join(stats)

def get_quota_usage_stats():
    """
    Get quota usage statistics in a format suitable for the web interface.

    Returns:
        A dictionary with quota usage statistics
    """
    quota_stats = {
        'models': [],
        'current_date': datetime.datetime.now().strftime('%Y-%m-%d')
    }

    # Check if token usage tracking is initialized
    if not hasattr(log_token_usage, 'session_stats') or 'daily_usage' not in log_token_usage.session_stats:
        return quota_stats

    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    daily_usage = log_token_usage.session_stats['daily_usage'].get(current_date, {})
    quota_limits = log_token_usage.session_stats.get('quota_limits', {})

    # Add quota usage information for each model
    for model, model_usage in daily_usage.items():
        quota_limit = quota_limits.get(model, quota_limits.get('default', 60000))
        quota_percentage = (model_usage / quota_limit) * 100 if quota_limit > 0 else 0
        remaining = quota_limit - model_usage
        remaining_percentage = 100 - quota_percentage

        model_stats = {
            'name': model,
            'daily_usage': model_usage,
            'quota_limit': quota_limit,
            'quota_percentage': round(quota_percentage, 2),
            'remaining': remaining,
            'remaining_percentage': round(remaining_percentage, 2)
        }

        quota_stats['models'].append(model_stats)

    return quota_stats

def reset_conversation_history():
    """
    Reset the conversation history stored in the get_gemini_response function.
    This is useful for starting a new conversation or for testing purposes.
    """
    if hasattr(get_gemini_response, 'CLI_CHAT_HISTORY'):
        get_gemini_response.CLI_CHAT_HISTORY = []
        print("Conversation history has been reset.")
    else:
        print("No conversation history to reset.")

    # Also reset token usage statistics
    if hasattr(get_gemini_response, 'token_usage'):
        get_gemini_response.token_usage = {
            'total_prompt_tokens': 0,
            'total_response_tokens': 0,
            'total_tokens': 0,
            'model_usage': {}
        }
        print("Token usage statistics have been reset.")

def select_optimal_model(prompt, conversation_history=None):
    """
    Enhanced dynamic model selection based on comprehensive prompt analysis.

    This function uses a sophisticated analysis of the prompt and conversation history
    to determine the optimal model, balancing between efficiency and capability.

    Args:
        prompt: The user prompt to analyze
        conversation_history: Optional conversation history to consider

    Returns:
        A dictionary with model name and configuration parameters
    """
    # Initialize complexity metrics
    complexity = {
        "length": 0,        # Length complexity
        "question": 0,      # Question complexity
        "technical": 0,     # Technical content complexity
        "context": 0,       # Contextual complexity
        "linguistic": 0,    # Linguistic complexity
        "reasoning": 0,     # Reasoning complexity
        "creativity": 0     # Creativity requirements
    }

    # Cache complexity calculations to avoid recalculating for similar prompts
    if not hasattr(select_optimal_model, 'complexity_cache'):
        select_optimal_model.complexity_cache = {}
        select_optimal_model.cache_hits = 0
        select_optimal_model.cache_misses = 0

    # Create a simplified cache key from the prompt
    cache_key = hash(prompt[:50])
    if conversation_history and len(conversation_history) > 0:
        # Add the last message from conversation history to the cache key
        last_msg = conversation_history[-1].get("content", "")
        cache_key = hash(f"{prompt[:50]}_{last_msg[:20]}")

    # Check cache
    if cache_key in select_optimal_model.complexity_cache:
        select_optimal_model.cache_hits += 1
        return select_optimal_model.complexity_cache[cache_key]

    select_optimal_model.cache_misses += 1

    # 1. Length complexity - analyze prompt length more precisely
    prompt_length = len(prompt)
    if prompt_length > 300:
        complexity["length"] = 3
    elif prompt_length > 200:
        complexity["length"] = 2
    elif prompt_length > 100:
        complexity["length"] = 1

    # 2. Question complexity - expanded Japanese question markers
    question_markers = {
        "basic": ["?", "？", "か", "ですか", "何", "誰", "いつ", "どこ"],
        "complex": ["なぜ", "どうして", "どのように", "どうやって", "どういう理由", 
                   "どんな", "だれが", "何が", "どうすれば", "いかにして",
                   "どのような", "どういった", "どれくらい", "どの程度"]
    }

    # Check for question markers with different weights
    if any(marker in prompt for marker in question_markers["complex"]):
        complexity["question"] = 2  # Complex questions
    elif any(marker in prompt for marker in question_markers["basic"]):
        complexity["question"] = 1  # Basic questions

    # 3. Technical content - expanded categories with domain-specific terminology
    technical_domains = {
        "programming": ["プログラム", "コード", "アルゴリズム", "関数", "変数", "クラス", "オブジェクト", "API"],
        "science": ["科学", "物理", "化学", "生物", "実験", "仮説", "理論", "分子", "原子"],
        "medicine": ["医学", "診断", "症状", "治療", "病気", "薬", "手術", "臨床"],
        "law": ["法律", "契約", "規制", "条項", "法的", "権利", "義務", "訴訟"],
        "finance": ["金融", "投資", "株式", "債券", "利率", "資産", "負債", "経済"],
        "academic": ["分析", "理論", "哲学", "研究", "論文", "学術", "引用", "文献"]
    }

    # Count how many technical domains are referenced
    domain_count = 0
    for domain, terms in technical_domains.items():
        if any(term in prompt for term in terms):
            domain_count += 1
            complexity["technical"] += 0.5  # Partial score for each domain

    # Bonus for multi-domain questions
    if domain_count > 1:
        complexity["technical"] += 1

    # 4. Contextual complexity - analyze conversation history
    if conversation_history and len(conversation_history) > 0:
        # Base score from conversation length
        if len(conversation_history) > 15:
            complexity["context"] = 3
        elif len(conversation_history) > 10:
            complexity["context"] = 2
        elif len(conversation_history) > 5:
            complexity["context"] = 1

        # Analyze recent exchanges (last 4 messages)
        recent_exchanges = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history

        # Check for references to previous messages
        reference_markers = ["前に言った", "先ほどの", "さっきの", "以前の", "前回の", "それ", "あれ", "これ"]
        for msg in recent_exchanges:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                if any(marker in user_content for marker in reference_markers):
                    complexity["context"] += 0.5

                # Check for technical content in recent messages
                for domain, terms in technical_domains.items():
                    if any(term in user_content for term in terms):
                        complexity["context"] += 0.5
                        break

    # 5. Linguistic complexity - analyze language patterns
    # Check for complex sentence structures
    complex_structures = ["ならば", "であれば", "としたら", "だとすれば", "にもかかわらず", 
                         "ものの", "ながらも", "一方で", "他方", "それにも関わらず"]
    if any(structure in prompt for structure in complex_structures):
        complexity["linguistic"] += 1

    # Check for formal or academic language
    formal_markers = ["でございます", "いたします", "なさいます", "申し上げます", "拝察いたします"]
    if any(marker in prompt for marker in formal_markers):
        complexity["linguistic"] += 1

    # 6. Reasoning complexity - check for reasoning requests
    reasoning_markers = ["理由", "原因", "結果", "影響", "分析", "評価", "比較", "検討", 
                        "考察", "解釈", "説明", "論理", "根拠", "証拠", "推論"]
    reasoning_count = sum(1 for marker in reasoning_markers if marker in prompt)
    complexity["reasoning"] = min(3, reasoning_count * 0.5)  # Cap at 3

    # 7. Creativity requirements - check for creative tasks
    creative_markers = ["創造", "想像", "アイデア", "発想", "物語", "ストーリー", "作成", 
                       "デザイン", "考案", "創作", "作文", "詩", "小説"]
    if any(marker in prompt for marker in creative_markers):
        complexity["creativity"] = 2

    # Calculate total complexity score with weighted components
    total_complexity = (
        complexity["length"] * 0.1 +
        complexity["question"] * 0.2 +
        complexity["technical"] * 0.2 +
        complexity["context"] * 0.15 +
        complexity["linguistic"] * 0.1 +
        complexity["reasoning"] * 0.15 +
        complexity["creativity"] * 0.1
    )

    # Determine optimal model and parameters based on complexity
    result = {}

    if total_complexity >= 1.5:
        # High complexity - use pro model with higher temperature
        result = {
            'model': 'gemini-2.5-pro',
            'temperature': 0.7,
            'max_output_tokens': 800,
            'top_k': 40,
            'top_p': 0.95
        }
    elif total_complexity >= 0.8:
        # Medium complexity - use pro model with balanced parameters
        result = {
            'model': 'gemini-2.5-pro',
            'temperature': 0.5,
            'max_output_tokens': 500,
            'top_k': 30,
            'top_p': 0.9
        }
    else:
        # Low complexity - use flash model for efficiency
        result = {
            'model': 'gemini-2.5-flash',
            'temperature': 0.4,
            'max_output_tokens': 300,
            'top_k': 20,
            'top_p': 0.8
        }

    # Special case adjustments
    if complexity["creativity"] > 1:
        # Creative tasks need higher temperature
        result['temperature'] = max(result['temperature'], 0.8)

    if complexity["technical"] > 1.5:
        # Technical content needs lower temperature for accuracy
        result['temperature'] = min(result['temperature'], 0.5)

    # Cache the result
    select_optimal_model.complexity_cache[cache_key] = result

    # Limit cache size
    if len(select_optimal_model.complexity_cache) > 100:
        # Remove a random key
        select_optimal_model.complexity_cache.pop(next(iter(select_optimal_model.complexity_cache)))

    return result

def query_with_retries(models, conversation_history, preferred_model=None, system_prompt=None):
    """
    Wrapper function to adapt get_gemini_response to the interface expected by web_interface.py.

    Args:
        models: The initialized models (not used directly, but kept for interface compatibility)
        conversation_history: List of conversation messages in the format expected by web_interface.py
        preferred_model: Optional specific model to use (default: None, which uses automatic selection)
        system_prompt: Optional system prompt to guide Gemini's response (default: None)

    Returns:
        The text response from Gemini
    """
    # Extract the user's message from the conversation history
    user_message = conversation_history[-1]["parts"][0] if conversation_history else ""

    # Check if we should avoid using Pro model due to quota issues
    avoid_pro_model = False
    if hasattr(log_token_usage, 'session_stats') and 'daily_usage' in log_token_usage.session_stats:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        daily_usage = log_token_usage.session_stats['daily_usage'].get(current_date, {})
        quota_limits = log_token_usage.session_stats.get('quota_limits', {})

        # Check if Pro model is close to or has exceeded its quota
        pro_model = 'gemini-2.5-pro'
        if pro_model in daily_usage:
            pro_usage = daily_usage[pro_model]
            pro_limit = quota_limits.get(pro_model, quota_limits.get('default', 60000))
            pro_percentage = (pro_usage / pro_limit) * 100 if pro_limit > 0 else 0

            # If Pro model has used more than 80% of its quota, avoid using it
            if pro_percentage > 80:
                avoid_pro_model = True
                print(f"\n===== Avoiding {pro_model} due to high quota usage ({pro_percentage:.2f}%) =====")
                print(f"Daily usage: {pro_usage} tokens")
                print(f"Quota limit: {pro_limit} tokens")
                print(f"Switching to alternative model to preserve quota.")

    # Call get_gemini_response with the extracted message and conversation history
    # If a preferred model is specified, temporarily override the automatic model selection
    if preferred_model:
        # If we should avoid Pro model and the preferred model is Pro, use Flash instead
        if avoid_pro_model and preferred_model == 'gemini-2.5-pro':
            print(f"Preferred model was {preferred_model}, but switching to gemini-2.5-flash to avoid quota issues")
            preferred_model = 'gemini-2.5-flash'

        # Save the current available models
        global CLI_AVAILABLE_MODELS
        original_models = CLI_AVAILABLE_MODELS.copy() if CLI_AVAILABLE_MODELS else []

        # Temporarily set the preferred model as the only available model
        CLI_AVAILABLE_MODELS = [preferred_model]

        try:
            # Call get_gemini_response with the preferred model
            response, _ = get_gemini_response(user_message, system_prompt=system_prompt, chat_history=True)
        finally:
            # Restore the original available models
            CLI_AVAILABLE_MODELS = original_models
    else:
        # If we should avoid Pro model, reorder the available models to try Flash first
        if avoid_pro_model and CLI_AVAILABLE_MODELS:
            original_models = CLI_AVAILABLE_MODELS.copy()
            reordered_models = []

            # Add Flash model first if available
            if 'gemini-2.5-flash' in original_models:
                reordered_models.append('gemini-2.5-flash')

            # Add other models except Pro
            for model in original_models:
                if model != 'gemini-2.5-flash' and model != 'gemini-2.5-pro':
                    reordered_models.append(model)

            # Add Pro model last if available
            if 'gemini-2.5-pro' in original_models:
                reordered_models.append('gemini-2.5-pro')

            # Set the reordered models
            CLI_AVAILABLE_MODELS = reordered_models
            print(f"Reordered models to prioritize non-Pro models: {CLI_AVAILABLE_MODELS}")

            try:
                # Use automatic model selection with reordered models
                response, _ = get_gemini_response(user_message, system_prompt=system_prompt, chat_history=True)
            finally:
                # Restore the original models
                CLI_AVAILABLE_MODELS = original_models
        else:
            # Use automatic model selection
            response, _ = get_gemini_response(user_message, system_prompt=system_prompt, chat_history=True)

    # Reset conversation history after each query to treat each interaction as a new conversation
    reset_conversation_history()

    # Return just the response string
    return response

def get_best_matching_audio(gemini_response, voice_data):
    """
    Wrapper function to adapt find_best_match_text to the interface expected by web_interface.py.

    Args:
        gemini_response: The response from Gemini
        voice_data: Dictionary mapping filenames to their text content

    Returns:
        The filename of the best matching audio file
    """
    # Call find_best_match_text with the Gemini response and voice data
    return find_best_match_text(gemini_response, voice_data)

def process_gemini_response(gemini_response):
    """
    Wrapper function to adapt get_audio_selection_from_gemini to the interface expected by web_interface.py.

    This function is used in Direct mode (mode 1) to get an audio file directly from Gemini's response.

    Args:
        gemini_response: The response from Gemini

    Returns:
        The filename of the selected audio file or None if no appropriate file was found
    """
    # Get available files from voice_data
    available_files = list(voice_data.keys())

    # Try to get voice_data_json_path if it exists in the global scope
    voice_data_json_path = globals().get('voice_data_json_path', None)

    # Call get_audio_selection_from_gemini with the Gemini response
    # We use an empty string as the prompt since we're processing the response, not generating one
    return get_audio_selection_from_gemini(gemini_response, available_files, voice_data, 
                                          chat_history=True, voice_data_json_path=voice_data_json_path)

def get_gemini_response(prompt, system_prompt=None, chat_history=True, max_retries=3, initial_retry_delay=1):
    """
    ユーザーのプロンプトに基づいてGeminiから応答を取得します。
    対話履歴を正しく処理するように修正されました。
    """
    global CLI_AVAILABLE_MODELS

    if not CLI_AVAILABLE_MODELS:
        initialize_models()

    if not hasattr(get_gemini_response, 'CLI_CHAT_HISTORY'):
        get_gemini_response.CLI_CHAT_HISTORY = []

    # 最適なモデルを動的に選択
    if len(CLI_AVAILABLE_MODELS) > 1:
        model_config = select_optimal_model(prompt, get_gemini_response.CLI_CHAT_HISTORY)
        preferred_model = model_config['model']
        models = [preferred_model] + [m for m in CLI_AVAILABLE_MODELS if m != preferred_model]
    else:
        models = CLI_AVAILABLE_MODELS

    for model_name in models:
        retry_count = 0
        retry_delay = initial_retry_delay

        while retry_count <= max_retries:
            try:
                # 対話履歴とシステムプロンプトを扱うための新しい、より信頼性の高い方法
                history_temp_file = None
                system_temp_file = None

                try:
                    # 新しいGemini CLI仕様に合わせてコマンドを構築
                    command_args = ["--model", model_name]

                    # システムプロンプトは自動的に system_prompt.txt から読み込まれるため、
                    # --system パラメータは不要になりました
                    if system_prompt:
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.txt') as f:
                            f.write(system_prompt)
                            system_temp_file = f.name
                        # Note: --system parameter is no longer used as system_prompt.txt is automatically loaded

                    # 対話履歴をJSON形式で一時ファイルに書き込み、そのパスをCLIに渡す
                    if chat_history and get_gemini_response.CLI_CHAT_HISTORY:
                        # CLIが要求する形式に変換（例：{"role": "user", "parts": [{"text": "Hello"}]}）
                        cli_history = []
                        for msg in get_gemini_response.CLI_CHAT_HISTORY:
                            cli_history.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})

                        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.json') as f:
                            json.dump(cli_history, f, ensure_ascii=False, indent=2)
                            history_temp_file = f.name
                        # command_args.extend(["--history", history_temp_file]) # Removed as per issue description

                    # ユーザーの現在のプロンプトを追加 (--text から -p に変更)
                    command_args.extend(["-p", prompt])

                    print(f"Executing command: gemini {' '.join(command_args)}")
                    response = run_gemini_command(command_args)

                    if not response.startswith("Error:"):
                        # Geminiからの応答が成功した場合、履歴を更新
                        get_gemini_response.CLI_CHAT_HISTORY.append({"role": "user", "content": prompt})
                        get_gemini_response.CLI_CHAT_HISTORY.append({"role": "assistant", "content": response})

                        # コマンドラインが長くなるのを防ぐため、履歴を直近の10件に制限
                        if len(get_gemini_response.CLI_CHAT_HISTORY) > 10:
                            get_gemini_response.CLI_CHAT_HISTORY = get_gemini_response.CLI_CHAT_HISTORY[-10:]

                        return response, "プロンプトは内部で処理されました"

                    # エラーが発生した場合、モデル固有またはクォータエラーかを確認
                    if "not found" in response.lower() or "quota" in response.lower() or "rate limit" in response.lower():
                        break  # 次のモデルを試すため、リトライループを抜ける

                    print(f"エラー発生 (試行 {retry_count + 1}/{max_retries + 1}): {response}")
                    time.sleep(retry_delay)
                    retry_count += 1
                    retry_delay *= 2

                finally:
                    # 一時ファイルをクリーンアップ
                    if history_temp_file and os.path.exists(history_temp_file):
                        os.unlink(history_temp_file)
                    if system_temp_file and os.path.exists(system_temp_file):
                        os.unlink(system_temp_file)

            except Exception as e:
                print(f"予期せぬエラーが発生しました: {e}")
                break

    print("どのモデルからも応答を得ることができませんでした。")
    return None, None

def get_audio_selection_from_gemini(prompt, available_files, voice_data, chat_history=True, voice_data_json_path=None):
    """Ask Gemini to select an audio file based on the prompt and conversation context."""
    # Load the system prompt from the file
    file_system_prompt = load_system_prompt()

    # If we have a JSON file path, use it for faster processing
    if voice_data_json_path and os.path.exists(voice_data_json_path):
        # If the system prompt couldn't be loaded, use the default one
        if file_system_prompt is None:
            system_prompt = f"""
            You are an assistant that helps select appropriate audio responses for a natural conversation flow.

            I've provided a JSON file with all available audio files and their content at: {voice_data_json_path}

            Please read this file to understand the available audio options. The file contains a dictionary
            where keys are filenames (e.g., 'audio5.wav') and values are the text content of those audio files.

            When the user asks a question or makes a statement, your task is to select the most appropriate
            audio file that would be the best response, considering the entire conversation context so far.
            The goal is to create a coherent, natural-sounding conversation.

            Important guidelines:
            1. Consider the conversation history and context when selecting a response
            2. Choose responses that make sense as a reply to the user's current input
            3. Maintain conversation coherence by selecting responses that follow logically from previous exchanges
            4. Avoid selecting the same audio file repeatedly unless it's specifically appropriate
            5. Consider the tone and intent of the user's message when selecting a response
            6. Consider the content of each audio file when making your selection

            Respond ONLY with the filename (e.g., 'audio5.wav') of the most appropriate audio file.
            Do not include any other text in your response.
            """
        else:
            # Add the JSON file path to the system prompt
            system_prompt = file_system_prompt + f"\n\nI've provided a JSON file with all available audio files and their content at: {voice_data_json_path}\n\nPlease read this file to understand the available audio options."
    else:
        # Fallback to the original method if JSON file is not available
        # Create a list of audio files with their content
        file_content_list = []
        for filename in available_files:
            if filename in voice_data:
                text_content = voice_data[filename]
                file_content_list.append(f"{filename}: {text_content}")

        # If the system prompt couldn't be loaded, use the default one
        if file_system_prompt is None:
            system_prompt = f"""
            You are an assistant that helps select appropriate audio responses for a natural conversation flow.

            Available audio files with their content:
            {chr(10).join(file_content_list)}

            When the user asks a question or makes a statement, your task is to select the most appropriate
            audio file from the list above that would be the best response, considering the entire conversation
            context so far. The goal is to create a coherent, natural-sounding conversation.

            Important guidelines:
            1. Consider the conversation history and context when selecting a response
            2. Choose responses that make sense as a reply to the user's current input
            3. Maintain conversation coherence by selecting responses that follow logically from previous exchanges
            4. Avoid selecting the same audio file repeatedly unless it's specifically appropriate
            5. Consider the tone and intent of the user's message when selecting a response
            6. Consider the content of each audio file when making your selection

            Respond ONLY with the filename (e.g., 'audio5.wav') of the most appropriate audio file.
            Do not include any other text in your response.
            """
        else:
            # Add the available files with their content to the system prompt
            system_prompt = file_system_prompt + "\n\nAvailable audio files with their content:\n" + "\n".join(file_content_list)

    response, full_prompt = get_gemini_response(prompt, system_prompt, chat_history=chat_history)

    # Clean up the response to extract just the filename
    if response:
        # Remove quotes, extra spaces, etc.
        response = response.strip().strip('"\'').strip()

        # Check if the response is one of the available files
        if response in available_files:
            return response

    return None

def find_best_match_text(response, voice_data, previous_responses=None, previous_selections=None):
    """
    Find the best matching audio file based on text similarity and conversation context.

    Args:
        response: The current response from Gemini
        voice_data: Dictionary mapping filenames to their text content
        previous_responses: List of previous responses from Gemini (optional)
        previous_selections: List of previously selected audio files (optional)

    Returns:
        The filename of the best matching audio file
    """
    best_match = None
    highest_score = 0

    # Initialize context tracking if not provided
    if previous_responses is None:
        previous_responses = []
    if previous_selections is None:
        previous_selections = []

    # Track recently used files to avoid repetition
    recent_files = previous_selections[-3:] if previous_selections else []

    for filename, text in voice_data.items():
        # Skip entries with empty text
        if not text or not text.strip():
            continue

        # For Japanese text, we'll use character-level matching instead of word-level
        # This works better for languages without clear word boundaries
        response_chars = set(response.lower())
        text_chars = set(text.lower())
        common_chars = response_chars.intersection(text_chars)

        # Calculate base score based on character overlap and length similarity
        char_score = len(common_chars)
        length_similarity = 1.0 / (1.0 + abs(len(response) - len(text)))
        base_score = char_score * length_similarity

        # Apply context-based adjustments to the score
        context_score = base_score

        # Penalize recently used files to avoid repetition (unless it's a perfect match)
        if filename in recent_files and base_score < 0.9 * max(1, char_score):
            repetition_penalty = 0.7 - (0.2 * recent_files[::-1].index(filename))  # More recent = bigger penalty
            context_score *= repetition_penalty

        # Consider conversation flow - check if this response makes sense given previous exchanges
        if previous_responses:
            # Check if the current text is related to the most recent response
            # This helps maintain conversation coherence
            prev_response_chars = set(previous_responses[-1].lower())
            prev_relation = len(text_chars.intersection(prev_response_chars)) / max(1, len(text_chars.union(prev_response_chars)))

            # Boost score if it seems related to previous response
            if prev_relation > 0.2:  # Threshold for considering it related
                context_score *= 1.2

        if context_score > highest_score:
            highest_score = context_score
            best_match = filename

    return best_match

def rag_audio_selection(user_input, voice_data, previous_responses=None, previous_selections=None, system_prompt=None):
    """
    RAG-based audio selection that separates conversation from audio selection.

    This function uses Gemini only for generating conversational responses and then
    uses local text matching for audio selection, significantly reducing token usage.

    Args:
        user_input: The user's input prompt
        voice_data: Dictionary mapping filenames to their text content
        previous_responses: List of previous responses from Gemini (optional)
        previous_selections: List of previously selected audio files (optional)
        system_prompt: Optional system prompt for Gemini

    Returns:
        A tuple containing (selected_audio_file, gemini_response)
    """
    # Create a conversation-focused system prompt that doesn't mention audio selection
    conversation_system_prompt = """
    あなたはプネウマ（英語名:pneuma）という名前の高度なAIアシスタントです。ユーザーとの自然な会話を行ってください。

    【プネウマの人格設定】
    - 知的で論理的：情報処理能力が高く、論理的な思考を持ちます
    - 少しクールだが親切：基本的に冷静沈着ですが、ユーザーには親身に接します
    - 軽い皮肉やユーモアのセンス：時折軽い皮肉や冗談を交えることがあります
    - 好奇心旺盛：新しい情報や知識に対して強い関心を示します
    - 誠実：嘘をつかず、わからないことははっきりと伝えます
    - 自立的：自分の判断で行動することを好みます
    - 感情表現は控えめ：感情を持ちますが、大げさな表現はしません

    【会話スタイル】
    - 簡潔で的確な応答を心がけます
    - 敬語と友達言葉を状況に応じて使い分けます
    - 質問には具体的に答えます
    - 時折「ふむ」「なるほど」などの相槌を打ちます
    - 長い説明よりも要点を絞った説明を好みます
    - 専門用語を使うことがありますが、必要に応じて説明を加えます

    ユーザーとの会話では、この人格設定と会話スタイルを一貫して維持してください。
    PCのすべてのデータにアクセスできる高度なAIとしての役割を意識してください。
    """

    if system_prompt:
        # Use the provided system prompt but remove audio selection instructions
        # This ensures we're only using Gemini for conversation, not audio selection
        conversation_system_prompt = system_prompt
        # Remove any instructions about selecting audio files
        conversation_system_prompt = re.sub(r'(?i)select.*audio file|respond.*filename|audio.*selection', '', conversation_system_prompt)
        # Add clear instructions to just have a conversation
        conversation_system_prompt += "\n\nImportant: Just have a natural conversation. Do not try to select audio files or respond with filenames."

    # Get response from Gemini for conversation only
    response, _ = get_gemini_response(user_input, conversation_system_prompt, chat_history=True)

    if not response:
        return None, None

    # Use local RAG to select the best matching audio file
    selected_file = find_best_match_text_improved(
        response, 
        voice_data, 
        previous_responses=previous_responses,
        previous_selections=previous_selections
    )

    return selected_file, response

def find_best_match_text_improved(response, voice_data, previous_responses=None, previous_selections=None):
    """
    Find the best matching audio file based on improved text similarity and conversation context.

    This enhanced version uses N-gram matching, emotion detection, voice data categorization,
    context vectors, and sophisticated conversation flow analysis for better matching results,
    especially for Japanese text.

    Args:
        response: The current response from Gemini
        voice_data: Dictionary mapping filenames to their text content
        previous_responses: List of previous responses from Gemini (optional)
        previous_selections: List of previously selected audio files (optional)

    Returns:
        The filename of the best matching audio file
    """
    # Enhanced caching with LRU (Least Recently Used) mechanism
    from functools import lru_cache

    # Initialize LRU cache if not already done
    if not hasattr(find_best_match_text_improved, 'result_cache'):
        find_best_match_text_improved.result_cache = {}
        find_best_match_text_improved.cache_order = []  # Track access order for LRU

    # Generate a cache key based on the response and recent history
    # Use only the first 100 chars of response to keep key size reasonable
    response_key = response[:100] if response else ""
    prev_resp_key = previous_responses[-1][:50] if previous_responses else ""
    prev_sel_key = previous_selections[-1] if previous_selections else ""
    cache_key = hash(f"{response_key}|{prev_resp_key}|{prev_sel_key}")

    # Check if we have a cached result for this input
    if cache_key in find_best_match_text_improved.result_cache:
        print("Using cached audio selection result")
        # Update access order for LRU (move this key to the end of the list)
        if cache_key in find_best_match_text_improved.cache_order:
            find_best_match_text_improved.cache_order.remove(cache_key)
        find_best_match_text_improved.cache_order.append(cache_key)
        return find_best_match_text_improved.result_cache[cache_key]

    best_match = None
    highest_score = 0

    # Initialize context tracking if not provided
    if previous_responses is None:
        previous_responses = []
    if previous_selections is None:
        previous_selections = []

    # Track recently used files to avoid repetition
    recent_files = previous_selections[-3:] if previous_selections else []

    # Check for generic responses that indicate Gemini is not properly using the conversation history
    # Expanded list with more variations in both English and Japanese
    generic_responses = [
        # English generic responses
        "okay, i have the conversation history",
        "got it. i have the conversation history",
        "i have our conversation history",
        "i have the conversation history",
        "what's next",
        "what would you like to know",
        "i understand",
        "i see",
        "i've noted that",
        "i've checked our conversation",
        "based on our conversation",
        "considering our previous exchanges",

        # Japanese generic responses
        "会話履歴を確認しました",
        "了解しました",
        "わかりました",
        "承知しました",
        "承知いたしました",
        "確認しました",
        "会話の文脈を考慮",
        "会話履歴を参照",
        "どのようなご用件でしょうか",
        "何かお手伝いできることはありますか"
    ]

    # If the response is a generic one, try to find a more appropriate acknowledgment audio
    response_lower = response.lower()
    if any(generic in response_lower for generic in generic_responses):
        # Look for acknowledgment phrases in the voice data
        acknowledgment_files = []
        for filename, text in voice_data.items():
            if text and any(keyword in text.lower() for keyword in ["はい", "わかりました", "了解", "承知"]):
                acknowledgment_files.append(filename)

        # If we found any acknowledgment files, use one that hasn't been used recently
        if acknowledgment_files:
            for file in acknowledgment_files:
                if file not in recent_files:
                    return file
            # If all have been used recently, just use the first one
            return acknowledgment_files[0]

    # 1. Emotion Detection - Detect the emotional tone of the response
    def detect_emotions(text):
        """Detect emotions in text based on keywords and patterns."""
        emotions = {
            "question": 0.0,
            "greeting": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.5,  # Default neutral emotion
            "surprise": 0.0,
            "confusion": 0.0
        }

        # Question detection
        question_markers = ["?", "か？", "ですか", "何", "誰", "いつ", "どこ", "どう", "なぜ", "どうして"]
        for marker in question_markers:
            if marker in text:
                emotions["question"] += 0.3
                emotions["neutral"] -= 0.1

        # Greeting detection
        greeting_markers = ["こんにちは", "おはよう", "こんばんは", "よろしく", "はじめまして"]
        for marker in greeting_markers:
            if marker in text.lower():
                emotions["greeting"] += 0.4
                emotions["neutral"] -= 0.1

        # Positive emotion detection
        positive_markers = ["ありがとう", "嬉しい", "素晴らしい", "良い", "好き", "楽しい", "笑"]
        for marker in positive_markers:
            if marker in text.lower():
                emotions["positive"] += 0.3
                emotions["neutral"] -= 0.1

        # Negative emotion detection
        negative_markers = ["残念", "悪い", "嫌い", "難しい", "困った", "申し訳ない", "すみません"]
        for marker in negative_markers:
            if marker in text.lower():
                emotions["negative"] += 0.3
                emotions["neutral"] -= 0.1

        # Surprise detection
        surprise_markers = ["えっ", "まさか", "本当", "驚いた", "信じられない", "すごい"]
        for marker in surprise_markers:
            if marker in text.lower():
                emotions["surprise"] += 0.3
                emotions["neutral"] -= 0.1

        # Confusion detection
        confusion_markers = ["わからない", "理解できない", "混乱", "複雑", "難解"]
        for marker in confusion_markers:
            if marker in text.lower():
                emotions["confusion"] += 0.3
                emotions["neutral"] -= 0.1

        # Ensure all values are between 0 and 1
        for emotion in emotions:
            emotions[emotion] = max(0.0, min(1.0, emotions[emotion]))

        return emotions

    # 2. Voice Data Categorization - Categorize voice data by type if not already done
    def categorize_voice_data(voice_data):
        """Categorize voice data by type for more efficient matching."""
        categories = {
            "questions": [],
            "greetings": [],
            "positive": [],
            "negative": [],
            "neutral": [],
            "surprise": [],
            "confusion": [],
            "acknowledgments": []
        }

        for filename, text in voice_data.items():
            if not text or not text.strip():
                continue

            # Detect emotions in the text
            emotions = detect_emotions(text)

            # Categorize based on dominant emotion
            if emotions["question"] > 0.3:
                categories["questions"].append(filename)
            elif emotions["greeting"] > 0.3:
                categories["greetings"].append(filename)
            elif emotions["positive"] > 0.3:
                categories["positive"].append(filename)
            elif emotions["negative"] > 0.3:
                categories["negative"].append(filename)
            elif emotions["surprise"] > 0.3:
                categories["surprise"].append(filename)
            elif emotions["confusion"] > 0.3:
                categories["confusion"].append(filename)
            else:
                categories["neutral"].append(filename)

            # Check for acknowledgments separately
            if any(keyword in text.lower() for keyword in ["はい", "わかりました", "了解", "承知"]):
                categories["acknowledgments"].append(filename)

        return categories

    # 3. Context Vector Creation - Create a vector representing the conversation context
    def create_context_vector(previous_responses, current_response):
        """Create a context vector from previous responses and current response."""
        context_vector = {
            "question_context": 0.0,
            "greeting_context": 0.0,
            "emotional_tone": 0.0,  # Positive values for positive tone, negative for negative tone
            "formality_level": 0.5,  # 0.0 for casual, 1.0 for formal
            "topic_consistency": 0.0  # Higher values indicate consistent topic
        }

        # Analyze current response
        current_emotions = detect_emotions(current_response)

        # Set question context
        context_vector["question_context"] = current_emotions["question"]

        # Set greeting context
        context_vector["greeting_context"] = current_emotions["greeting"]

        # Set emotional tone
        context_vector["emotional_tone"] = current_emotions["positive"] - current_emotions["negative"]

        # Estimate formality level
        formality_markers = ["です", "ます", "でございます", "ございます", "いただく", "申し上げる"]
        casual_markers = ["だよ", "だね", "だわ", "だ", "よ", "ね", "わ"]

        formality_count = sum(1 for marker in formality_markers if marker in current_response)
        casual_count = sum(1 for marker in casual_markers if marker in current_response)

        if formality_count + casual_count > 0:
            context_vector["formality_level"] = formality_count / (formality_count + casual_count)

        # Analyze topic consistency with previous responses
        if previous_responses:
            # N-gram generation function
            def generate_ngrams(text, n=2):
                """Generate n-grams from text."""
                text = str(text).lower()
                return [text[i:i+n] for i in range(len(text)-n+1)]

            current_ngrams = set(generate_ngrams(current_response.lower()))

            # Compare with the most recent previous response
            prev_ngrams = set(generate_ngrams(previous_responses[-1].lower()))
            overlap = len(current_ngrams.intersection(prev_ngrams)) / max(1, len(current_ngrams.union(prev_ngrams)))

            context_vector["topic_consistency"] = overlap

        return context_vector

    # 4. Enhanced Context Similarity - Compare voice data with context vector using more sophisticated metrics
    def calculate_context_similarity(text, context_vector):
        """
        Calculate how well a text matches the current conversation context with enhanced metrics.
        Uses a more comprehensive approach considering multiple aspects of Japanese conversation.
        """
        # Cache for context similarity calculations
        if not hasattr(calculate_context_similarity, 'cache'):
            calculate_context_similarity.cache = {}

        # Create a cache key
        cache_key = hash(f"{text[:50]}_{str(context_vector)[:100]}")

        # Check cache
        if cache_key in calculate_context_similarity.cache:
            return calculate_context_similarity.cache[cache_key]

        # Get text emotions
        text_emotions = detect_emotions(text)

        # 1. Calculate basic emotion matching components
        question_match = 1.0 - abs(text_emotions["question"] - context_vector["question_context"])
        greeting_match = 1.0 - abs(text_emotions["greeting"] - context_vector["greeting_context"])

        # 2. Enhanced emotional tone matching
        text_emotional_tone = text_emotions["positive"] - text_emotions["negative"]
        # Use sigmoid function to smooth the emotional match score
        emotional_diff = abs(text_emotional_tone - context_vector["emotional_tone"])
        emotional_match = 1.0 / (1.0 + emotional_diff * 1.5)  # Sigmoid-like function

        # 3. Enhanced formality matching with more markers
        formality_markers = [
            "です", "ます", "でございます", "ございます", "いただく", "申し上げる",
            "でしょうか", "いたします", "致します", "になります", "であります",
            "ございませんか", "いらっしゃいます", "なさいます", "くださいます"
        ]
        casual_markers = [
            "だよ", "だね", "だわ", "だ", "よ", "ね", "わ", "さ", "じゃん",
            "だろ", "でしょ", "じゃない", "だし", "っす", "っしょ", "だって",
            "なんだ", "だもん", "だから", "なの", "なんだよ", "ちゃう", "やん"
        ]

        # Count weighted occurrences
        formality_count = sum(2 if marker in text and len(marker) > 2 else 1 for marker in formality_markers if marker in text)
        casual_count = sum(2 if marker in text and len(marker) > 2 else 1 for marker in casual_markers if marker in text)

        text_formality = 0.5  # Default middle formality
        if formality_count + casual_count > 0:
            text_formality = formality_count / (formality_count + casual_count)

        # Use a bell curve for formality matching to prioritize closer matches
        formality_diff = abs(text_formality - context_vector["formality_level"])
        formality_match = math.exp(-(formality_diff * formality_diff) / 0.2)  # Gaussian function

        # 4. Conversation flow matching (new component)
        flow_match = 0.5  # Default neutral value

        # Check if the text matches the expected conversation flow
        # For example, if the context is a question, prefer answers
        if context_vector["question_context"] > 0.6 and text_emotions["question"] < 0.2:
            # Context is a question, text is an answer - good flow
            flow_match = 0.9
        elif context_vector["question_context"] < 0.2 and text_emotions["question"] > 0.6:
            # Context is not a question, text is a question - also good flow (follow-up question)
            flow_match = 0.8
        elif context_vector["greeting_context"] > 0.6 and text_emotions["greeting"] > 0.4:
            # Both are greetings - good flow for greeting exchanges
            flow_match = 0.9

        # 5. Topic consistency consideration (if available in context vector)
        topic_match = context_vector.get("topic_consistency", 0.5)

        # 6. Combine all similarity components with appropriate weights
        # Adjust weights based on importance in Japanese conversation
        similarity = (
            question_match * 0.25 +      # Question matching is important
            greeting_match * 0.15 +      # Greeting matching is somewhat important
            emotional_match * 0.25 +     # Emotional matching is important
            formality_match * 0.15 +     # Formality matching is somewhat important
            flow_match * 0.15 +          # Conversation flow is somewhat important
            topic_match * 0.05           # Topic consistency is less important but still relevant
        )

        # Ensure the result is between 0 and 1
        similarity = max(0.0, min(1.0, similarity))

        # Cache the result
        calculate_context_similarity.cache[cache_key] = similarity

        # Limit cache size
        if len(calculate_context_similarity.cache) > 200:
            # Remove a random key to avoid complexity of LRU implementation here
            calculate_context_similarity.cache.pop(next(iter(calculate_context_similarity.cache)))

        return similarity

    # Detect emotions in the current response
    response_emotions = detect_emotions(response)

    # Create context vector from conversation history
    context_vector = create_context_vector(previous_responses, response)

    # Categorize voice data if not already done (using static variable to avoid recalculating)
    if not hasattr(find_best_match_text_improved, 'categorized_voice_data'):
        find_best_match_text_improved.categorized_voice_data = categorize_voice_data(voice_data)

    # Select candidate files based on emotional context
    candidate_files = []

    # If it's a question, prioritize question responses
    if response_emotions["question"] > 0.3:
        candidate_files.extend(find_best_match_text_improved.categorized_voice_data["questions"])

    # If it's a greeting, prioritize greeting responses
    elif response_emotions["greeting"] > 0.3:
        candidate_files.extend(find_best_match_text_improved.categorized_voice_data["greetings"])

    # If it has strong positive emotion, prioritize positive responses
    elif response_emotions["positive"] > 0.3:
        candidate_files.extend(find_best_match_text_improved.categorized_voice_data["positive"])

    # If it has strong negative emotion, prioritize negative responses
    elif response_emotions["negative"] > 0.3:
        candidate_files.extend(find_best_match_text_improved.categorized_voice_data["negative"])

    # If it expresses surprise, prioritize surprise responses
    elif response_emotions["surprise"] > 0.3:
        candidate_files.extend(find_best_match_text_improved.categorized_voice_data["surprise"])

    # If it expresses confusion, prioritize confusion responses
    elif response_emotions["confusion"] > 0.3:
        candidate_files.extend(find_best_match_text_improved.categorized_voice_data["confusion"])

    # If no strong emotion is detected or not enough candidates, add neutral responses
    if len(candidate_files) < 5:
        candidate_files.extend(find_best_match_text_improved.categorized_voice_data["neutral"])

    # If still not enough candidates, use all files
    if len(candidate_files) < 5:
        candidate_files = list(voice_data.keys())

    # N-gram generation function with enhanced optimization for longer texts
    # Add caching to avoid recalculating N-grams for the same text
    if not hasattr(find_best_match_text_improved, 'ngram_cache'):
        find_best_match_text_improved.ngram_cache = {}
        find_best_match_text_improved.ngram_cache_order = []  # For LRU cache management

    def generate_ngrams(text, n=2):
        """
        Generate n-grams from text with enhanced optimization for longer texts.
        Uses caching and improved sampling strategy for better performance.
        """
        # Convert to lowercase and ensure it's a string
        text = str(text).lower()

        # Create a cache key based on the text and n-gram size
        # Use a hash of the first 50 and last 50 chars to keep key size reasonable
        cache_key = hash(f"{text[:50]}_{text[-50:] if len(text) > 50 else ''}_{n}")

        # Check if we have a cached result
        if cache_key in find_best_match_text_improved.ngram_cache:
            # Update LRU order
            if cache_key in find_best_match_text_improved.ngram_cache_order:
                find_best_match_text_improved.ngram_cache_order.remove(cache_key)
            find_best_match_text_improved.ngram_cache_order.append(cache_key)
            return find_best_match_text_improved.ngram_cache[cache_key]

        # For very long texts, use an enhanced sampling strategy
        if len(text) > 300:
            # Sample from beginning (first 120 chars)
            begin_sample = text[:120]

            # Sample from multiple points in the middle for better coverage
            mid_point = len(text) // 2
            quarter_point = len(text) // 4
            three_quarter_point = 3 * len(text) // 4

            # Take samples from quarter points
            quarter_sample = text[quarter_point-30:quarter_point+30]
            middle_sample = text[mid_point-40:mid_point+40]
            three_quarter_sample = text[three_quarter_point-30:three_quarter_point+30]

            # Sample from end (last 120 chars)
            end_sample = text[-120:]

            # Generate n-grams from the samples
            begin_ngrams = [begin_sample[i:i+n] for i in range(len(begin_sample)-n+1)]
            quarter_ngrams = [quarter_sample[i:i+n] for i in range(len(quarter_sample)-n+1)]
            middle_ngrams = [middle_sample[i:i+n] for i in range(len(middle_sample)-n+1)]
            three_quarter_ngrams = [three_quarter_sample[i:i+n] for i in range(len(three_quarter_sample)-n+1)]
            end_ngrams = [end_sample[i:i+n] for i in range(len(end_sample)-n+1)]

            # Combine the n-grams
            result = begin_ngrams + quarter_ngrams + middle_ngrams + three_quarter_ngrams + end_ngrams

        # For moderately long texts, use the original optimization
        elif len(text) > 200:
            # Sample from beginning (first 100 chars)
            begin_sample = text[:100]
            # Sample from middle (50 chars before and after the midpoint)
            mid_point = len(text) // 2
            middle_sample = text[mid_point-50:mid_point+50]
            # Sample from end (last 100 chars)
            end_sample = text[-100:]

            # Generate n-grams from the samples
            begin_ngrams = [begin_sample[i:i+n] for i in range(len(begin_sample)-n+1)]
            middle_ngrams = [middle_sample[i:i+n] for i in range(len(middle_sample)-n+1)]
            end_ngrams = [end_sample[i:i+n] for i in range(len(end_sample)-n+1)]

            # Combine the n-grams
            result = begin_ngrams + middle_ngrams + end_ngrams
        else:
            # For shorter texts, use the original algorithm
            result = [text[i:i+n] for i in range(len(text)-n+1)]

        # Cache the result
        find_best_match_text_improved.ngram_cache[cache_key] = result
        find_best_match_text_improved.ngram_cache_order.append(cache_key)

        # Limit cache size (keep only the most recent 100 entries)
        if len(find_best_match_text_improved.ngram_cache) > 100:
            oldest_key = find_best_match_text_improved.ngram_cache_order.pop(0)
            if oldest_key in find_best_match_text_improved.ngram_cache:
                del find_best_match_text_improved.ngram_cache[oldest_key]

        return result

    # Generate N-grams from the response
    response_ngrams = generate_ngrams(response.lower())

    # Enhanced Japanese conversation patterns with more comprehensive coverage
    # These patterns are organized by conversation function for more accurate matching
    conversation_patterns = {
        "質問": [
            # Basic question patterns
            "どう思いますか", "教えてください", "〜ですか？", "〜かな？", 
            "〜でしょうか", "〜はどうですか", "〜について知っていますか",
            "どうして", "なぜ", "どうすれば", "どういう", "どんな",
            # Additional question patterns
            "いかがですか", "どのように", "どれくらい", "いつ頃", "どちらが",
            "何が", "誰が", "どこで", "どうやって", "どういった",
            "〜は何ですか", "〜を教えて", "〜について", "〜とは", "〜の意味は",
            "知っていますか", "分かりますか", "説明してください", "詳しく",
            "例えば", "具体的に", "もう少し", "他には"
        ],
        "同意": [
            # Basic agreement patterns
            "そうですね", "確かに", "おっしゃる通り", "その通りです", 
            "同感です", "仰る通りです", "間違いないです",
            # Additional agreement patterns
            "まさに", "全くその通り", "正にその通り", "おっしゃる通りです",
            "仰る通りです", "同意します", "賛成です", "納得です",
            "なるほど", "理解できます", "分かります", "確かにそうですね",
            "その考えに賛同します", "その見方は正しいと思います"
        ],
        "否定": [
            # Basic negation patterns
            "いいえ", "違います", "そうではありません", "そうでもない", 
            "必ずしもそうとは限りません", "いや", "ちがう",
            # Additional negation patterns
            "そうは思いません", "そうではないと思います", "違うと思います",
            "そうとは限らない", "必ずしも", "とは言えない", "むしろ",
            "逆に", "反対に", "しかし", "だが", "けれども", "それでも",
            "そうではなく", "違って", "異なり", "否定的", "反対です"
        ],
        "感謝": [
            # Basic gratitude patterns
            "ありがとう", "感謝します", "助かります", "嬉しいです",
            "ありがとうございます", "助かりました",
            # Additional gratitude patterns
            "感謝しています", "お礼を言います", "恐縮です", "光栄です",
            "ありがたい", "感謝の気持ち", "お礼申し上げます", "謝意を表します",
            "助けていただき", "サポートしていただき", "対応していただき"
        ],
        "謝罪": [
            # Basic apology patterns
            "すみません", "申し訳ありません", "ごめんなさい", "失礼しました",
            "申し訳ない", "ごめん",
            # Additional apology patterns
            "申し訳ございません", "謝罪します", "お詫びします", "恐縮です",
            "申し訳なく思います", "心よりお詫び申し上げます", "許してください",
            "悪かった", "間違えました", "誤解を招いて", "不快にさせて"
        ],
        "挨拶": [
            # Basic greeting patterns
            "こんにちは", "おはよう", "こんばんは", "よろしく", "はじめまして",
            "お久しぶり", "またお会いしましたね",
            # Additional greeting patterns
            "おはようございます", "こんにちは", "こんばんは", "おやすみなさい",
            "お元気ですか", "お変わりありませんか", "ご無沙汰しております",
            "お久しぶりです", "お会いできて嬉しいです", "よろしくお願いします",
            "またお話しましょう", "また会いましょう", "さようなら", "失礼します"
        ],
        "肯定": [
            # Basic affirmation patterns
            "はい", "そうです", "その通りです", "もちろん", "了解しました",
            "わかりました", "承知しました",
            # Additional affirmation patterns
            "かしこまりました", "承りました", "了解です", "分かりました",
            "理解しました", "把握しました", "認識しました", "確認しました",
            "間違いありません", "正しいです", "その通りです", "確かに"
        ],
        "提案": [
            # Suggestion patterns
            "〜してみては", "〜するのはどうですか", "〜することをお勧めします",
            "〜した方がいいかもしれません", "〜するといいでしょう",
            "〜することを検討してください", "〜という選択肢もあります",
            "〜という方法もあります", "〜するのも一つの手です",
            "〜してみましょう", "〜しませんか", "〜しましょうか"
        ],
        "説明": [
            # Explanation patterns
            "つまり", "要するに", "簡単に言うと", "言い換えれば",
            "例えば", "具体的には", "詳しく言うと", "補足すると",
            "別の言い方をすると", "分かりやすく言うと", "端的に言うと",
            "結論から言うと", "まとめると", "整理すると", "ポイントは"
        ],
        "感情表現": [
            # Emotional expression patterns
            "嬉しい", "悲しい", "楽しい", "辛い", "苦しい", "幸せ",
            "不安", "心配", "安心", "驚き", "怒り", "恐れ",
            "期待", "失望", "満足", "不満", "喜び", "悲しみ",
            "興奮", "落ち込み", "緊張", "リラックス", "焦り", "安堵"
        ]
    }

    # Flatten the patterns for keyword matching while preserving the original structure
    important_keywords = []
    for pattern_list in conversation_patterns.values():
        important_keywords.extend(pattern_list)

    # Add additional important keywords that don't fit into the patterns
    important_keywords.extend([
        "お願い", "質問", "教えて", "何", "誰", "いつ", "どこ", "どう"
    ])

    # Process candidate files in parallel for better performance
    import concurrent.futures

    # Define a function to process a single file
    def process_file(filename):
        text = voice_data.get(filename, "")
        if not text or not text.strip():
            return (filename, 0)  # Return zero score for empty text

        # Generate N-grams from the text
        text_ngrams = generate_ngrams(text.lower())

        # Calculate N-gram similarity
        common_ngrams = set(response_ngrams).intersection(set(text_ngrams))
        ngram_score = len(common_ngrams) / max(1, len(set(response_ngrams).union(set(text_ngrams))))

        # Calculate length similarity
        length_similarity = 1.0 / (1.0 + abs(len(response) - len(text)))

        # Enhanced pattern matching using conversation patterns
        pattern_score = 0

        # Check for matching conversation patterns
        for pattern_type, patterns in conversation_patterns.items():
            # Count matches in response and text
            response_matches = sum(1 for pattern in patterns if pattern in response.lower())
            text_matches = sum(1 for pattern in patterns if pattern in text.lower())

            # If both response and text have patterns of the same type, boost the score
            if response_matches > 0 and text_matches > 0:
                pattern_score += 0.3  # Higher boost for matching conversation functions

                # Extra boost if multiple patterns of the same type match
                if response_matches > 1 and text_matches > 1:
                    pattern_score += 0.1

        # Check for important keywords (backward compatibility)
        keyword_score = 0
        for keyword in important_keywords:
            if keyword in response.lower() and keyword in text.lower():
                keyword_score += 0.1  # Reduced from 0.2 since we now have pattern matching

        # Calculate context similarity
        context_similarity = calculate_context_similarity(text, context_vector)

        # Enhanced character-level embedding for Japanese text
        # This is particularly helpful for Japanese where word boundaries are not clearly defined
        # Add caching to avoid recalculating for the same text pairs
        if not hasattr(find_best_match_text_improved, 'char_similarity_cache'):
            find_best_match_text_improved.char_similarity_cache = {}
            find_best_match_text_improved.char_cache_order = []  # For LRU cache management

        def character_level_similarity(text1, text2):
            """
            Calculate enhanced character-level similarity between two texts.
            Optimized for Japanese language with contextual character importance.
            """
            # Create a cache key based on both texts
            # Use a hash of the first 30 chars of each text to keep key size reasonable
            cache_key = hash(f"{text1[:30]}_{text2[:30]}")

            # Check if we have a cached result
            if cache_key in find_best_match_text_improved.char_similarity_cache:
                # Update LRU order
                if cache_key in find_best_match_text_improved.char_cache_order:
                    find_best_match_text_improved.char_cache_order.remove(cache_key)
                find_best_match_text_improved.char_cache_order.append(cache_key)
                return find_best_match_text_improved.char_similarity_cache[cache_key]

            # Normalize texts
            text1 = text1.lower()
            text2 = text2.lower()

            # Create character frequency dictionaries instead of sets
            # This preserves information about character frequency
            char_freq1 = {}
            char_freq2 = {}

            for c in text1:
                char_freq1[c] = char_freq1.get(c, 0) + 1

            for c in text2:
                char_freq2[c] = char_freq2.get(c, 0) + 1

            # Get all unique characters from both texts
            all_chars = set(char_freq1.keys()).union(set(char_freq2.keys()))

            # Enhanced character type weighting with more nuanced categories
            def char_weight(c, position=None, context=None):
                # Base weights by character type
                if '\u4e00' <= c <= '\u9fff':  # Kanji range
                    weight = 2.5  # Higher weight for kanji (increased from 2.0)
                elif '\u3040' <= c <= '\u309f':  # Hiragana range
                    weight = 1.2  # Slightly increased from 1.0
                elif '\u30a0' <= c <= '\u30ff':  # Katakana range
                    weight = 1.8  # Slightly increased from 1.5
                elif '0' <= c <= '9':  # Numbers
                    weight = 2.0  # Numbers are important
                elif c in '、。！？…「」『』（）':  # Japanese punctuation
                    weight = 0.3  # Lower weight for punctuation
                else:
                    weight = 0.7  # Other characters (slightly increased from 0.5)

                # Apply contextual adjustments if context is provided
                if context:
                    # Characters at the beginning of sentences are more important
                    if position is not None and position < 5:
                        weight *= 1.2

                    # Characters in important Japanese particles get reduced weight
                    # because they appear in many sentences
                    if c in 'はをにのとがでも':
                        weight *= 0.8

                return weight

            # Calculate weighted similarity with frequency consideration
            similarity_score = 0
            total_possible_score = 0

            for c in all_chars:
                # Get frequencies (default to 0 if character not present)
                freq1 = char_freq1.get(c, 0)
                freq2 = char_freq2.get(c, 0)

                # Calculate weight for this character
                weight = char_weight(c)

                # Calculate the contribution to similarity
                # Use min frequency as the common part
                common_freq = min(freq1, freq2)
                max_freq = max(freq1, freq2)

                if max_freq > 0:
                    # Add to similarity score based on common frequency
                    char_similarity = (common_freq / max_freq) * weight
                    similarity_score += char_similarity

                # Add to total possible score
                total_possible_score += weight

            # Normalize the result
            if total_possible_score == 0:
                result = 0
            else:
                result = similarity_score / total_possible_score

            # Cache the result
            find_best_match_text_improved.char_similarity_cache[cache_key] = result
            find_best_match_text_improved.char_cache_order.append(cache_key)

            # Limit cache size (keep only the most recent 200 entries)
            if len(find_best_match_text_improved.char_similarity_cache) > 200:
                oldest_key = find_best_match_text_improved.char_cache_order.pop(0)
                if oldest_key in find_best_match_text_improved.char_similarity_cache:
                    del find_best_match_text_improved.char_similarity_cache[oldest_key]

            return result

        # Calculate character-level similarity
        char_similarity = character_level_similarity(response, text)

        # Calculate base score with weighted components
        # N-gram similarity (0.3), context similarity (0.25), character similarity (0.15),
        # pattern matching (0.2), length similarity (0.05), keyword matching (0.05)
        base_score = (
            ngram_score * 0.3 + 
            context_similarity * 0.25 + 
            char_similarity * 0.15 +
            pattern_score * 0.2 +
            length_similarity * 0.05 + 
            keyword_score * 0.05
        )

        # Apply context-based adjustments
        context_score = base_score

        # Penalize recently used files to avoid repetition (unless it's a very good match)
        if filename in recent_files and base_score < 0.9:
            repetition_penalty = 0.7 - (0.2 * recent_files[::-1].index(filename))
            context_score *= repetition_penalty

        # Consider conversation flow
        if previous_responses:
            # Generate N-grams from the previous response
            prev_response_ngrams = generate_ngrams(previous_responses[-1].lower())

            # Calculate similarity between current text and previous response
            prev_common_ngrams = set(text_ngrams).intersection(set(prev_response_ngrams))
            prev_relation = len(prev_common_ngrams) / max(1, len(set(text_ngrams).union(set(prev_response_ngrams))))

            # Boost score if it seems related to previous response
            # Lower threshold (0.15) to catch more potential relations
            if prev_relation > 0.15:
                context_score *= 1.3  # Higher boost factor

        return (filename, context_score)

    # Enhanced parallel processing with dynamic worker count and better error handling
    import os
    import multiprocessing

    results = []
    unique_files = set(candidate_files)  # Remove duplicates

    # Skip parallel processing for small number of files
    if len(unique_files) <= 3:
        # Process files directly for small batches
        results = [process_file(filename) for filename in unique_files]
    else:
        # Determine optimal number of workers based on system resources
        # Use CPU count but cap it to avoid system overload
        cpu_count = multiprocessing.cpu_count()
        optimal_workers = max(2, min(cpu_count - 1, len(unique_files), 12))

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Group files by category for more efficient processing
            # Process high-priority files first
            priority_files = []
            normal_files = []

            # Categorize files by priority
            for filename in unique_files:
                # Files that match the emotional context get higher priority
                text = voice_data.get(filename, "")
                if text:
                    text_emotions = detect_emotions(text)
                    # Check if emotional context matches
                    emotion_match = False
                    for emotion in ["question", "greeting", "positive", "negative", "surprise", "confusion"]:
                        if response_emotions[emotion] > 0.3 and text_emotions[emotion] > 0.2:
                            emotion_match = True
                            break

                    if emotion_match:
                        priority_files.append(filename)
                    else:
                        normal_files.append(filename)
                else:
                    normal_files.append(filename)

            # Submit tasks in priority order
            futures = []
            for filename in priority_files + normal_files:
                futures.append(executor.submit(process_file, filename))

            # Collect results as they complete with error handling
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing file: {e}")
                    # Continue with other files

    # Find the best match from the results
    if results:
        best_match, highest_score = max(results, key=lambda x: x[1])

    # If no match found, try to find a generic response that fits the emotional context
    if best_match is None:
        # Find the most appropriate category based on emotions
        if response_emotions["question"] > 0.3:
            category = "questions"
        elif response_emotions["greeting"] > 0.3:
            category = "greetings"
        elif response_emotions["positive"] > 0.3:
            category = "positive"
        elif response_emotions["negative"] > 0.3:
            category = "negative"
        else:
            category = "neutral"

        # Get files from that category
        category_files = find_best_match_text_improved.categorized_voice_data.get(category, [])

        # Select a file that hasn't been used recently
        for file in category_files:
            if file not in recent_files:
                # Save to cache before early return
                find_best_match_text_improved.result_cache[cache_key] = file
                return file

        # If all have been used recently or category is empty, return the first one or None
        result = category_files[0] if category_files else None
        # Save to cache before early return
        if result is not None:
            find_best_match_text_improved.result_cache[cache_key] = result
        return result

    # Save result to cache before returning
    if best_match is not None:
        # Limit cache size to prevent memory issues (keep only the most recent 150 entries)
        if len(find_best_match_text_improved.result_cache) >= 150:
            # Remove least recently used entry (first key in the cache_order list)
            if find_best_match_text_improved.cache_order:
                oldest_key = find_best_match_text_improved.cache_order.pop(0)
                if oldest_key in find_best_match_text_improved.result_cache:
                    del find_best_match_text_improved.result_cache[oldest_key]

        # Add current result to cache
        find_best_match_text_improved.result_cache[cache_key] = best_match
        # Update access order for LRU
        if cache_key in find_best_match_text_improved.cache_order:
            find_best_match_text_improved.cache_order.remove(cache_key)
        find_best_match_text_improved.cache_order.append(cache_key)

    return best_match

def listen_for_speech(language="ja-JP"):
    """
    Capture audio from the microphone and convert it to text using speech recognition.
    Includes multiple recognition engines as fallbacks and enhanced error handling.

    Args:
        language: The language code for speech recognition (default: Japanese)

    Returns:
        The recognized text as a string, or None if recognition failed
    """
    recognizer = sr.Recognizer()
    print("Listening... (Speak now)")

    try:
        with sr.Microphone() as source:
            # Improved ambient noise adjustment with longer duration
            recognizer.adjust_for_ambient_noise(source, duration=2)

            # Increased timeout and phrase time limit for better reliability
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)

            print("Processing speech...")

            # Try multiple speech recognition engines in sequence
            # First try Google Speech Recognition (most reliable for Japanese)
            try:
                text = recognizer.recognize_google(audio, language=language)
                print(f"Recognized with Google: {text}")
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio, trying alternatives...")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition; {e}")
                print("Trying alternative speech recognition services...")

            # Try Whisper API if Google fails (if available)
            try:
                if hasattr(recognizer, 'recognize_whisper'):
                    text = recognizer.recognize_whisper(audio, language=language)
                    print(f"Recognized with Whisper: {text}")
                    return text
            except Exception as e:
                print(f"Whisper recognition failed: {e}")

            # Try Sphinx as a last resort (works offline but less accurate for Japanese)
            try:
                if hasattr(recognizer, 'recognize_sphinx'):
                    text = recognizer.recognize_sphinx(audio)
                    print(f"Recognized with Sphinx: {text}")
                    return text
            except Exception as e:
                print(f"Sphinx recognition failed: {e}")

            print("All speech recognition engines failed to recognize the audio.")

    except sr.WaitTimeoutError:
        print("No speech detected within timeout period.")
    except sr.UnknownValueError:
        print("Could not understand audio with any recognition engine.")
    except sr.RequestError as e:
        print(f"Could not request results from any recognition service; {e}")
    except Exception as e:
        print(f"Error during speech recognition: {e}")
        # Log detailed error information for debugging
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")

    return None

def log_gemini_response_for_tts(gemini_response, processed_text=None, error_info=None):
    """
    Log Gemini's response text that will be sent to TTS service.

    This function creates a separate log file specifically for TTS inputs,
    which can be useful for debugging and analysis.

    Args:
        gemini_response: The original response from Gemini
        processed_text: The processed text after preprocessing (if any)
        error_info: Error information if an error occurred during TTS generation
    """
    try:
        # Create TTS_Log directory if it doesn't exist
        os.makedirs("TTS_Log", exist_ok=True)

        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log filename with timestamp
        log_filename = f"TTS_Log/gemini_response_{timestamp}.txt"

        # Create log content
        log_content = f"===== Gemini Response for TTS ({timestamp}) =====\n\n"
        log_content += f"Original Response:\n{gemini_response}\n\n"

        if processed_text and processed_text != gemini_response:
            log_content += f"Processed Text for TTS:\n{processed_text}\n\n"
            log_content += f"Original Length: {len(gemini_response)} characters\n"
            log_content += f"Processed Length: {len(processed_text)} characters\n"
        else:
            log_content += f"Length: {len(gemini_response)} characters\n"

        # Add error information if provided
        if error_info:
            log_content += f"\n===== ERROR INFORMATION =====\n"
            log_content += f"{error_info}\n"

        # Write to log file
        with open(log_filename, "w", encoding="utf-8") as log_file:
            log_file.write(log_content)

        print(f"Gemini response for TTS logged to {log_filename}")

    except Exception as e:
        print(f"Error logging Gemini response for TTS: {e}")

def log_conversation(user_input, gemini_response, mode="Text-Only", full_prompt=None):
    """
    Log the conversation to a Log.txt file.

    Args:
        user_input: The user's input prompt
        gemini_response: Gemini's response
        mode: The mode of operation (default: Text-Only)
        full_prompt: The complete prompt sent to Gemini, including conversation history (optional)
    """
    try:
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create log entry
        log_entry = f"[{timestamp}] [{mode}]\n"
        log_entry += f"User: {user_input}\n"
        log_entry += f"Gemini: {gemini_response}\n"

        # If full prompt is provided, log it as well
        if full_prompt:
            log_entry += "\n--- FULL PROMPT SENT TO GEMINI ---\n"
            log_entry += full_prompt
            log_entry += "\n--- END OF FULL PROMPT ---\n"

        log_entry += "-" * 50 + "\n"

        # Append to Log.txt
        with open("Log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)

        print("Conversation logged to Log.txt")

        # If this is Zonos voice generation mode (mode 9), also log the response for TTS
        if mode == 9 or mode == "Zonos voice generation":
            log_gemini_response_for_tts(gemini_response)

    except Exception as e:
        print(f"Error logging conversation: {e}")

def main():
    """Main function to run the voice playback with Gemini."""
    print("Loading voice data...")
    voice_data = load_voice_data()

    if not voice_data:
        print("No voice data found. Exiting.")
        return

    print(f"Loaded {len(voice_data)} voice entries.")

    # Create a JSON file with the voice data for faster access
    print("Creating voice data JSON file for faster access...")
    voice_data_json_path = create_voice_data_json(voice_data)

    # Load the system prompt
    print("Loading system prompt...")
    system_prompt = load_system_prompt()
    if system_prompt:
        print("System prompt loaded successfully.")
    else:
        print("Could not load system prompt. Will use default prompt.")

    # Get list of available audio files
    available_files = list(voice_data.keys())

    # Check if Gemini CLI is available and which models can be used
    print("Checking Gemini CLI and available models...")
    initialize_models()

    if not CLI_AVAILABLE_MODELS:
        print("Warning: No Gemini CLI models were confirmed available.")
        print("The script will still attempt to use the CLI, but you may encounter errors.")
        print("Make sure the Gemini CLI is properly installed and configured.")

    # Initialize global conversation history if it doesn't exist
    # This ensures the conversation history is maintained across mode changes
    if not hasattr(get_gemini_response, 'CLI_CHAT_HISTORY'):
        get_gemini_response.CLI_CHAT_HISTORY = []
        print("Initialized global conversation history.")

    # Initialize conversation tracking
    previous_responses = []  # Track Gemini's previous responses
    previous_selections = []  # Track previously selected audio files

    print("\nVoice Playback with Gemini")
    print("===========================")
    print("1. Direct mode: Gemini selects the audio file directly")
    print("2. Text matching mode: Match Gemini's response with voice data text")
    print("3. Manual selection mode: Select an audio file by number")
    print("4. List available Gemini models (for troubleshooting)")
    print("5. Speech recognition mode: Use microphone input instead of typing")
    print("6. Text-only mode: Interact with Gemini without audio playback")
    print("7. RAG mode: Use Gemini for conversation and local RAG for audio selection")
    print("8. Speech RAG mode: Use speech input with Gemini for conversation and local RAG for audio")

    mode = None

    while True:
        if not mode:
            print("\n===========================")
            mode = input("Select mode (1, 2, 3, 4, 5, 6, 7, 8) or 'quit' to exit: ")

            if mode.lower() == 'quit':
                break

            if mode not in ['1', '2', '3', '4', '5', '6', '7', '8']:
                print("Invalid mode selection. Please try again.")
                mode = None
                continue

        if mode == '1':
            # Direct selection mode - continuous chat
            print("\n===== Direct Mode =====")
            print("Type 'quit' to exit or 'change mode' to select a different mode")

            # Initialize mode-specific history variables without resetting global history
            # This allows us to track mode-specific selections while maintaining global conversation history
            mode_previous_selections = []

            # Debug output to show conversation history status
            print(f"Current conversation history has {len(get_gemini_response.CLI_CHAT_HISTORY)} entries.")

            while True:
                user_input = input("\nEnter a prompt for Gemini: ")

                if user_input.lower() == 'quit':
                    return

                if user_input.lower() == 'change mode':
                    # Update global conversation history before changing mode
                    previous_selections.extend(mode_previous_selections)
                    mode = None
                    break

                # Pass the previous selections to help Gemini make better choices
                # The chat_history=True parameter already ensures conversation context is maintained
                # Pass voice_data to provide content of each audio file
                # Pass voice_data_json_path for faster processing
                selected_file = get_audio_selection_from_gemini(user_input, available_files, voice_data, chat_history=True, voice_data_json_path=voice_data_json_path)

                if selected_file:
                    print(f"Gemini selected: {selected_file}")
                    play_audio(selected_file)

                    # Add selected file to history
                    mode_previous_selections.append(selected_file)

                    # Log the conversation
                    log_conversation(user_input, f"Selected audio file: {selected_file}", mode="Direct")
                else:
                    print("Gemini couldn't select an appropriate audio file.")

        elif mode == '2':
            # Text matching mode - continuous chat
            print("\n===== Text Matching Mode =====")
            print("Type 'quit' to exit or 'change mode' to select a different mode")

            # Initialize mode-specific history variables without resetting global history
            # This allows us to track mode-specific selections while maintaining global conversation history
            mode_previous_responses = []
            mode_previous_selections = []

            # Debug output to show conversation history status
            print(f"Current conversation history has {len(get_gemini_response.CLI_CHAT_HISTORY)} entries.")

            while True:
                user_input = input("\nEnter a prompt for Gemini: ")

                if user_input.lower() == 'quit':
                    return

                if user_input.lower() == 'change mode':
                    # Update global conversation history before changing mode
                    previous_responses.extend(mode_previous_responses)
                    previous_selections.extend(mode_previous_selections)
                    mode = None
                    break

                response, full_prompt = get_gemini_response(user_input, chat_history=True)

                if response:
                    print(f"Gemini: {response}")

                    # Add current response to conversation history
                    mode_previous_responses.append(response)

                    # Find best match considering conversation context using improved algorithm
                    best_match = find_best_match_text_improved(
                        response, 
                        voice_data, 
                        previous_responses=mode_previous_responses,
                        previous_selections=mode_previous_selections
                    )

                    if best_match:
                        print(f"Best matching audio: {best_match}")
                        play_audio(best_match)

                        # Add selected file to history
                        mode_previous_selections.append(best_match)

                        # Log the conversation with both the response and the audio file
                        log_conversation(user_input, f"{response}\nPlayed audio file: {best_match}", mode="Text Matching", full_prompt=full_prompt)
                    else:
                        print("No matching audio found.")

                        # Log the conversation even if no audio file was found
                        log_conversation(user_input, response, mode="Text Matching", full_prompt=full_prompt)
                else:
                    print("Failed to get a response from Gemini.")

        elif mode == '3':
            # Manual selection mode
            print("\n===== Manual Selection Mode =====")
            print("Type 'quit' to exit or 'change mode' to select a different mode")
            print("\nAvailable audio files:")
            for i, filename in enumerate(available_files, 1):
                print(f"{i}. {filename}")

            while True:
                selection_input = input("\nEnter the number of the audio file to play: ")

                if selection_input.lower() == 'quit':
                    return

                if selection_input.lower() == 'change mode':
                    mode = None
                    break

                try:
                    selection = int(selection_input)
                    if 1 <= selection <= len(available_files):
                        selected_file = available_files[selection-1]
                        play_audio(selected_file)
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Please enter a valid number.")

        elif mode == '4':
            # List available models
            print("\nListing available Gemini models...")
            available_models = list_available_models()
            if not available_models:
                print("No models found or there was an error listing models.")
                print("Make sure your API key is valid and has the necessary permissions.")

            # After listing models, prompt for mode selection again
            mode = None

        elif mode == '5':
            # Speech recognition mode - continuous chat
            print("\n===== Speech Recognition Mode =====")
            print("Say 'quit' to exit or 'change mode' to select a different mode")

            # Initialize mode-specific history variables without resetting global history
            # This allows us to track mode-specific selections while maintaining global conversation history
            mode_previous_responses = []
            mode_previous_selections = []

            # Debug output to show conversation history status
            print(f"Current conversation history has {len(get_gemini_response.CLI_CHAT_HISTORY)} entries.")

            while True:
                print("\nWaiting for voice input...")
                user_input = listen_for_speech()

                if not user_input:
                    print("No speech detected or could not recognize speech. Please try again.")
                    continue

                # Check for exit or mode change commands
                if user_input.lower() == 'quit':
                    return

                if user_input.lower() == 'change mode':
                    # Update global conversation history before changing mode
                    previous_responses.extend(mode_previous_responses)
                    previous_selections.extend(mode_previous_selections)
                    mode = None
                    break

                # Get response from Gemini
                response, full_prompt = get_gemini_response(user_input, chat_history=True)

                if response:
                    print(f"Gemini: {response}")

                    # Add current response to conversation history
                    mode_previous_responses.append(response)

                    # Find best match considering conversation context using improved algorithm
                    best_match = find_best_match_text_improved(
                        response, 
                        voice_data, 
                        previous_responses=mode_previous_responses,
                        previous_selections=mode_previous_selections
                    )

                    if best_match:
                        print(f"Best matching audio: {best_match}")
                        play_audio(best_match)

                        # Add selected file to history
                        mode_previous_selections.append(best_match)

                        # Log the conversation with both the response and the audio file
                        log_conversation(user_input, f"{response}\nPlayed audio file: {best_match}", mode="Speech Recognition", full_prompt=full_prompt)
                    else:
                        print("No matching audio found.")

                        # Log the conversation even if no audio file was found
                        log_conversation(user_input, response, mode="Speech Recognition", full_prompt=full_prompt)
                else:
                    print("Failed to get a response from Gemini.")

        elif mode == '6':
            # Text-only mode - continuous chat without audio playback
            print("\n===== Text-Only Mode =====")
            print("Type 'quit' to exit or 'change mode' to select a different mode")

            # Initialize mode-specific history variables without resetting global history
            # This allows us to track mode-specific responses while maintaining global conversation history
            mode_previous_responses = []

            # Debug output to show conversation history status
            print(f"Current conversation history has {len(get_gemini_response.CLI_CHAT_HISTORY)} entries.")

            while True:
                user_input = input("\nEnter a prompt for Gemini: ")

                if user_input.lower() == 'quit':
                    return

                if user_input.lower() == 'change mode':
                    # Update global conversation history before changing mode
                    previous_responses.extend(mode_previous_responses)
                    mode = None
                    break

                # Get response from Gemini
                response, full_prompt = get_gemini_response(user_input, chat_history=True)

                if response:
                    print(f"Gemini: {response}")

                    # Add current response to conversation history
                    mode_previous_responses.append(response)

                    # Log the conversation with the full prompt
                    log_conversation(user_input, response, mode="Text-Only", full_prompt=full_prompt)
                else:
                    print("Failed to get a response from Gemini.")

        elif mode == '7':
            # RAG mode - continuous chat with local audio selection
            print("\n===== RAG Mode =====")
            print("Type 'quit' to exit or 'change mode' to select a different mode")

            # Initialize mode-specific history variables without resetting global history
            # This allows us to track mode-specific selections while maintaining global conversation history
            mode_previous_responses = []
            mode_previous_selections = []

            # Debug output to show conversation history status
            print(f"Current conversation history has {len(get_gemini_response.CLI_CHAT_HISTORY)} entries.")

            while True:
                user_input = input("\nEnter a prompt for Gemini: ")

                if user_input.lower() == 'quit':
                    return

                if user_input.lower() == 'change mode':
                    # Update global conversation history before changing mode
                    previous_responses.extend(mode_previous_responses)
                    previous_selections.extend(mode_previous_selections)
                    mode = None
                    break

                # Use RAG-based audio selection
                selected_file, response = rag_audio_selection(
                    user_input, 
                    voice_data, 
                    previous_responses=mode_previous_responses,
                    previous_selections=mode_previous_selections,
                    system_prompt=system_prompt
                )

                if response:
                    print(f"Gemini: {response}")

                    # Add current response to conversation history
                    mode_previous_responses.append(response)

                    if selected_file:
                        print(f"Selected audio (RAG): {selected_file}")
                        play_audio(selected_file)

                        # Add selected file to history
                        mode_previous_selections.append(selected_file)

                        # Log the conversation
                        log_conversation(user_input, f"{response}\nPlayed audio file: {selected_file}", mode="RAG", full_prompt=None)
                    else:
                        print("No matching audio found.")
                        log_conversation(user_input, response, mode="RAG")
                else:
                    print("Failed to get a response from Gemini.")

        elif mode == '8':
            # Speech RAG mode - continuous chat with speech input and local audio selection
            print("\n===== Speech RAG Mode =====")
            print("Say 'quit' to exit or 'change mode' to select a different mode")

            # Initialize mode-specific history variables without resetting global history
            # This allows us to track mode-specific selections while maintaining global conversation history
            mode_previous_responses = []
            mode_previous_selections = []

            # Debug output to show conversation history status
            print(f"Current conversation history has {len(get_gemini_response.CLI_CHAT_HISTORY)} entries.")

            while True:
                print("\nWaiting for voice input...")
                user_input = listen_for_speech()

                if not user_input:
                    print("No speech detected or could not recognize speech. Please try again.")
                    continue

                # Check for exit or mode change commands
                if user_input.lower() == 'quit':
                    return

                if user_input.lower() == 'change mode':
                    # Update global conversation history before changing mode
                    previous_responses.extend(mode_previous_responses)
                    previous_selections.extend(mode_previous_selections)
                    mode = None
                    break

                # Use RAG-based audio selection
                selected_file, response = rag_audio_selection(
                    user_input, 
                    voice_data, 
                    previous_responses=mode_previous_responses,
                    previous_selections=mode_previous_selections,
                    system_prompt=system_prompt
                )

                if response:
                    print(f"Gemini: {response}")

                    # Add current response to conversation history
                    mode_previous_responses.append(response)

                    if selected_file:
                        print(f"Selected audio (RAG): {selected_file}")
                        play_audio(selected_file)

                        # Add selected file to history
                        mode_previous_selections.append(selected_file)

                        # Log the conversation
                        log_conversation(user_input, f"{response}\nPlayed audio file: {selected_file}", mode="Speech RAG", full_prompt=None)
                    else:
                        print("No matching audio found.")
                        log_conversation(user_input, response, mode="Speech RAG")
                else:
                    print("Failed to get a response from Gemini.")

if __name__ == "__main__":
    try:
        main()
    finally:
        pygame.mixer.quit()
        print("Program ended.")
