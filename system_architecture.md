# VBeing_Research System Architecture

## Overview

This document provides a comprehensive overview of the VBeing_Research system architecture, which integrates Gemini AI models with voice playback capabilities to create an interactive AI assistant named "Pneuma".

## System Components

The system consists of the following main components:

1. **Web Interface** (`web_interface.py`)
   - Flask-based web application
   - Provides user interface for interaction
   - Manages conversation history
   - Handles API requests

2. **Core Processing Engine** (`play_voice_with_gemini.py`)
   - Interfaces with Gemini CLI
   - Processes user inputs
   - Manages audio selection and playback
   - Handles conversation context

3. **Voice Data Management**
   - Voice files stored in `Voice/` directory
   - Voice metadata stored in `voiceData.csv`
   - Clone voices stored in `Voice/CloneVoice/` directory

4. **External Services**
   - Gemini API (via CLI)
   - Zonos API (for voice generation)
   - Speech recognition services

5. **Configuration**
   - System prompt (`system_prompt.txt`)
   - Environment variables (`.env`)
   - System prompt templates (`system_prompt_templates/`)

## System Architecture Diagram

```
+----------------------------------+
|        User's Web Browser        |
+------------------+---------------+
                   |
                   v
+----------------------------------+
|      Web Interface (Flask)       |
|        web_interface.py          |
+------------------+---------------+
                   |
                   v
+----------------------------------+
|    Core Processing Engine        |
|    play_voice_with_gemini.py     |
+-+----------------+---------------+
  |                |
  v                v
+--------+    +-------------------+
| Voice  |    |  External APIs    |
| Data   |    |                   |
+--------+    | +---------------+ |
| .wav   |    | | Gemini CLI    | |
| .webm  |<-->| +---------------+ |
| files  |    | | Zonos API     | |
|        |    | +---------------+ |
| CSV    |    | | Speech        | |
| data   |    | | Recognition   | |
+--------+    | +---------------+ |
              +-------------------+
```

## Data Flow

1. **User Input Flow**
   - User enters text or speaks into the web interface
   - Web interface sends request to Flask backend
   - Flask backend processes request and calls appropriate functions in the core engine
   - Core engine generates response using Gemini API
   - Response is sent back to web interface and displayed to user

2. **Audio Response Flow**
   - Based on the selected mode, the system either:
     - Selects an appropriate audio file from existing voice data (Modes 1, 2, 5, 7, 8)
     - Generates new audio using Zonos API (Mode 9)
   - Audio is played through the web interface
   - Audio files and metadata are stored for future use

3. **Configuration Flow**
   - System prompt guides Gemini's responses
   - Environment variables configure API connections
   - System prompt templates provide mode-specific instructions

## Operation Modes

The system supports multiple operation modes:

1. **Direct Mode (1)**: Gemini selects the audio file directly
2. **Text Matching Mode (2)**: Match Gemini's response with voice data text
3. **Manual Selection Mode (3)**: Select an audio file by number
4. **List Available Models (4)**: Display all available Gemini models
5. **Speech Recognition Mode (5)**: Use microphone input instead of typing
6. **Text-Only Mode (6)**: Interact with Gemini without audio playback
7. **RAG Mode (7)**: Use Gemini for conversation and local RAG for audio selection
8. **Speech RAG Mode (8)**: Use speech input with Gemini for conversation and local RAG for audio
9. **Zonos Voice Generation Mode (9)**: Generate voice responses using Zonos API

## Error Handling and Resilience

The system includes several mechanisms for error handling and resilience:

1. **Quota Management**
   - Tracks API usage to prevent quota exhaustion
   - Falls back to alternative models when quotas are exceeded
   - Provides clear error messages about quota limitations

2. **Connection Retries**
   - Implements exponential backoff for transient errors
   - Caches connection status to reduce unnecessary API calls

3. **Response Filtering**
   - Detects and filters out log entries and generic acknowledgments
   - Ensures meaningful responses to user queries

## Security Considerations

1. **API Authentication**
   - Gemini CLI uses system authentication
   - Zonos API key stored in environment variables

2. **Data Storage**
   - Voice data stored locally
   - Conversation logs stored in `Log.txt`

## Future Enhancements

Potential areas for future enhancement include:

1. Improved voice cloning capabilities
2. Enhanced RAG implementation for more contextual responses
3. Multi-language support
4. Integration with additional AI models and voice services
