# Gemini Voice Interface - Web GUI

This is a web-based graphical user interface for the Gemini Voice Interaction System. It allows you to interact with Gemini through a browser interface, with support for text and speech input, and audio playback.

## Features

- Clean, responsive web interface
- Support for all major modes from the original CLI application:
  - Direct mode: Gemini selects the audio file directly
  - Text matching mode: Match Gemini's response with voice data text
  - Speech recognition mode: Use your microphone for input
  - Text-only mode: Interact with Gemini without audio playback
  - RAG mode: Token-efficient conversation with local audio selection
  - Speech RAG mode: Speech input with token-efficient processing
- Real-time conversation display
- Audio playback of selected responses
- Placeholder for future 3D character integration

## Requirements

- All requirements from the original CLI application
- Flask (for the web server)
- A modern web browser with JavaScript enabled
- For speech recognition: A browser that supports the Web Speech API (Chrome, Edge, Safari, etc.)

## Setup

1. Follow all the setup instructions from the original README.md
2. Install Flask:
   ```
   pip install flask
   ```
3. Run the web interface:
   ```
   python web_interface.py
   ```
4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. When the page loads, the system will initialize automatically
2. Select a mode by clicking one of the mode buttons
3. For text input:
   - Type your message in the input field
   - Press Enter or click the Send button
4. For speech input (in modes 5 and 8):
   - Click the microphone button
   - Speak your message
   - The system will automatically process your speech when you stop talking
5. Gemini's responses will appear in the chat area
6. If audio playback is enabled for the selected mode, you'll hear the corresponding audio file

## Modes

### Mode 1: Direct Selection
Gemini will directly select which audio file to play based on your input.

### Mode 2: Text Matching
The system matches Gemini's response with the text in voiceData.csv to find the most appropriate audio file.

### Mode 5: Speech Recognition
Speak into your microphone instead of typing. The system will convert your speech to text and process it.

### Mode 6: Text-Only
Interact with Gemini through text only, without any audio playback.

### Mode 7: RAG Mode
Uses Gemini only for generating conversational responses and then uses local text matching for audio selection, reducing token usage.

### Mode 8: Speech RAG Mode
Combines speech recognition with token-efficient RAG processing.

## Future Enhancements

### 3D Character Integration

The web interface has been prepared for future 3D character integration:

- A placeholder in the UI for the character
- A canvas element for WebGL rendering
- JavaScript hooks for character animations during audio playback
- WebGL support detection
- Directory structure for 3D models at `static/models/`

To implement 3D characters in the future:

1. Add 3D model files to the `static/models/` directory
2. Update the `static/character.js` file with actual rendering code
3. The interface will automatically animate the character when audio is playing

### Other Planned Enhancements

- More customization options for the interface
- Support for additional languages
- Improved audio handling and visualization

## Troubleshooting

- If you encounter issues with the web interface, check the browser console for error messages
- For speech recognition issues, ensure your browser supports the Web Speech API and that you've granted microphone permissions
- If audio doesn't play, check that your browser's autoplay settings allow audio playback
- For other issues, refer to the troubleshooting section in the original README.md

## Integration with the Original CLI

The web interface uses the same core functionality as the original CLI application, maintaining all the optimizations and features while providing a more accessible and visual interface.
