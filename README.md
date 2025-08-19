# Voice Playback with Gemini

This project uses Google's Gemini AI to play audio files from the `Voice/` directory according to the data in `voiceData.csv`.

## Overview

The script provides five modes of operation:

1. **Direct mode**: Gemini selects the audio file directly based on your prompt
2. **Text matching mode**: Match Gemini's response with voice data text
3. **Manual selection mode**: Select an audio file by number
4. **List available models**: Display all available Gemini models (for troubleshooting)
5. **Speech recognition mode**: Use microphone input instead of typing

## Requirements

- Python 3.7+
- pygame library (for audio playback)
- google-generativeai library (for Gemini API)
- speech_recognition library (for speech recognition)
- pyaudio library (for microphone access)

## Setup

1. An API key is already configured in the script. If you need to use your own:
   - Get a Gemini API key from [Google AI Studio](https://ai.google.dev/)
   - Open `play_voice_with_gemini.py` and replace the existing API key with your own
2. Install required packages:
   ```
   pip install pygame google-generativeai SpeechRecognition pyaudio
   ```

## Usage

1. Run the script:
   ```
   python play_voice_with_gemini.py
   ```
2. Select a mode (1, 2, 3, 4, or 5) once at the beginning
3. Continue the conversation with Gemini in a continuous chat session
4. Type or say 'quit' to exit or 'change mode' to select a different mode

### Mode 1: Direct Selection

In this mode, you provide a prompt to Gemini, and it directly selects which audio file to play based on the available files. The conversation history is maintained, so Gemini remembers previous interactions for more contextual responses.

### Mode 2: Text Matching

In this mode, you provide a prompt to Gemini, and the script matches Gemini's response with the text in `voiceData.csv` to find the most appropriate audio file to play. The conversation history is maintained, allowing for more natural and contextual interactions.

### Mode 3: Manual Selection

In this mode, you can directly select which audio file to play from a numbered list. You can continue selecting files without returning to the mode selection menu.

### Mode 4: List Available Models

In this mode, the script will display all available Gemini models that can be used with your API key. This is useful for troubleshooting when you encounter model-related errors or want to check which models are accessible to you.

### Mode 5: Speech Recognition

In this mode, you can speak into your microphone instead of typing. The script will capture your voice, convert it to text using Google's Speech Recognition service, and then process it with Gemini just like in Mode 2 (Text Matching). This allows for a more natural, hands-free interaction. The default language is set to Japanese (ja-JP), but this can be modified in the code if needed.

## Performance Optimizations

The script includes several optimizations for faster response times:

1. **Model Initialization**: Models are initialized once at startup rather than for each request
2. **Conversation History**: The script maintains conversation history for more contextual responses
3. **Continuous Chat**: You only need to select a mode once, then can continue chatting without interruption
4. **Chat Sessions**: Uses Gemini's chat sessions for more efficient API usage

## File Structure

- `play_voice_with_gemini.py`: The main Python script
- `voiceData.csv`: CSV file mapping audio filenames to their text content
- `Voice/`: Directory containing WAV audio files

## Troubleshooting

- If you encounter encoding issues with Japanese text, the script will try different encodings (utf-8, shift-jis, euc-jp, iso-2022-jp)
- Make sure your Gemini API key is valid and has not expired
- Ensure all required packages are installed
- The script uses the 'gemini-1.5-flash' model with fallback to 'gemini-1.5-pro' and 'gemini-pro' if needed
- If you encounter quota limit errors (429), the script will:
  1. Attempt a single retry with a short delay (1 second)
  2. Quickly fall back to alternative models if the primary model's quota is exhausted
  3. Provide clear error messages about quota limitations
- Free tier API keys have limited quota. If you're frequently hitting limits, consider:
  1. Spacing out your requests
  2. Upgrading to a paid API tier for higher quotas
  3. Creating a new API key if your current one has exhausted its quota
