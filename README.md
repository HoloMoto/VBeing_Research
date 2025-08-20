# Voice Playback with Gemini

[日本語版はこちら (Japanese Version)](README_JP.md)

This project uses Google's Gemini AI to play audio files from the `Voice/` directory according to the data in `voiceData.csv`.

## Overview

The script provides eight modes of operation:

1. **Direct mode**: Gemini selects the audio file directly based on your prompt
2. **Text matching mode**: Match Gemini's response with voice data text
3. **Manual selection mode**: Select an audio file by number
4. **List available models**: Display all available Gemini models (for troubleshooting)
5. **Speech recognition mode**: Use microphone input instead of typing
6. **Text-only mode**: Interact with Gemini without audio playback
7. **RAG mode**: Use Gemini for conversation and local RAG for audio selection (token-efficient)
8. **Speech RAG mode**: Use speech input with Gemini for conversation and local RAG for audio (token-efficient)

All conversations with Gemini are automatically logged to a `Log.txt` file for future reference.

## Requirements

- Python 3.7+
- pygame library (for audio playback)
- speech_recognition library (for speech recognition)
- pyaudio library (for microphone access)
- python-dotenv library (for loading environment variables)
- Gemini CLI installed and configured on your system

## Setup

1. Install the Gemini CLI on your system
   - Follow the [official Gemini CLI installation instructions](https://cloud.google.com/gemini/docs/codeassist/gemini-cli)
   - Make sure you've authenticated with the CLI before running this script
2. Configure the path to the Gemini CLI in the .env file:
   - Add `GEMINI_CLI_PATH=<path_to_gemini_cli>` to your .env file
   - Example: `GEMINI_CLI_PATH=C:/Users/username/AppData/Roaming/npm/gemini.cmd`
3. (Optional) Configure a custom system prompt:
   - Create a text file with your system prompt instructions
   - Add `SYSTEM_PROMPT_PATH=<path_to_system_prompt_file>` to your .env file
   - Example: `SYSTEM_PROMPT_PATH=system_prompt.txt`
4. Install required Python packages:
   ```
   pip install pygame SpeechRecognition pyaudio python-dotenv
   ```

## Usage

1. Run the script:
   ```
   python play_voice_with_gemini.py
   ```
2. Select a mode (1, 2, 3, 4, 5, 6, 7, or 8) once at the beginning
3. Continue the conversation with Gemini in a continuous chat session
4. Type or say 'quit' to exit or 'change mode' to select a different mode
5. All conversations are automatically logged to `Log.txt` in the same directory

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

### Mode 6: Text-Only

In this mode, you interact with Gemini through text only, without any audio playback. This is useful for general conversations with Gemini or when you don't need audio responses. All conversations in this mode are logged to the `Log.txt` file for future reference.

### Mode 7: RAG Mode

In this mode, the script uses Gemini only for generating conversational responses and then uses local text matching (Retrieval-Augmented Generation) for audio selection. This significantly reduces token usage compared to Mode 1 or Mode 2, as Gemini is not asked to select audio files directly. The conversation history is maintained for contextual responses, and the local RAG algorithm selects the most appropriate audio file based on Gemini's response.

### Mode 8: Speech RAG Mode

This mode combines the token efficiency of Mode 7 with the convenience of speech recognition from Mode 5. You can speak into your microphone, and the script will convert your speech to text, use Gemini to generate a response, and then use local RAG to select the most appropriate audio file. This provides a fully voice-interactive experience while minimizing API token usage.

## Conversation Logging

All interactions with Gemini are automatically logged to a `Log.txt` file in the same directory as the script. Each log entry includes:

- Timestamp of the conversation
- Mode used for the interaction
- User's input prompt
- Gemini's response
- Audio file played (if applicable)

This logging feature allows you to review past conversations, track Gemini's responses over time, and maintain a record of all interactions. The log file is created automatically if it doesn't exist and new entries are appended to the end of the file.

## Performance Optimizations

The script includes numerous optimizations for faster response times, reduced token consumption, and improved accuracy:

1. **Model Initialization**: Models are initialized once at startup rather than for each request
2. **Conversation History**: The script maintains conversation history by including previous exchanges in the prompt, creating more contextual and coherent conversations
3. **Continuous Chat**: You only need to select a mode once, then can continue chatting without interruption
4. **Efficient Prompt Management**: Uses ultra-concise prompt formatting to minimize token usage while maintaining context
5. **Local File Access**: Uses a local JSON file to store voice data, allowing Gemini to read the file directly instead of including all voice data in each prompt, significantly reducing response time
6. **Token-Efficient Modes**: RAG modes (7 and 8) use Gemini only for conversation and handle audio selection locally, dramatically reducing token usage
7. **Optimized Error Handling**: Immediately tries the next model on quota errors instead of retrying, saving tokens
8. **Model Prioritization**: Uses gemini-2.5-flash as the default model, which has higher quota limits than gemini-2.5-pro

### Enhanced Caching and Performance
9. **LRU Caching System**: Implements Least Recently Used (LRU) caching for audio selection, N-gram calculation, character similarity, and complexity analysis to dramatically improve response speed
10. **Multi-level Caching**: Uses specialized caching for different components with appropriate cache sizes and eviction policies
11. **Optimized N-gram Calculation**: Enhanced algorithm that samples strategically from beginning, quarter-points, middle, and end of longer texts to reduce computation while improving accuracy
12. **Dynamic Parallel Processing**: Automatically adjusts thread count based on CPU cores and workload size, with priority-based task scheduling for critical paths

### Improved Japanese Language Processing
13. **Comprehensive Japanese Pattern Matching**: Expanded pattern recognition with 10+ categories and 150+ patterns covering various conversation functions (questions, agreements, negations, greetings, etc.)
14. **Context-aware Similarity Calculation**: Uses sophisticated metrics including emotional tone matching, formality matching, conversation flow analysis, and topic consistency
15. **Advanced Character-Level Embedding**: Frequency-based character analysis with weighted importance by character type (kanji, hiragana, katakana, punctuation) and position
16. **Japanese Particle Optimization**: Special handling for Japanese particles and sentence structures to improve matching accuracy

### Intelligent Resource Management
17. **Dynamic Conversation History**: Automatically adjusts history length and detail based on conversation complexity and context references
18. **Sophisticated Model Selection**: Analyzes 7 different complexity metrics (length, question type, technical content, context, linguistic complexity, reasoning, creativity) to select optimal model and parameters
19. **Parameter Optimization**: Dynamically adjusts temperature, max output tokens, and other parameters based on conversation needs
20. **Adaptive Processing**: Skips expensive operations for simple inputs and applies full processing only when needed

### Token Optimization and Monitoring
21. **Enhanced Token Estimation**: Highly accurate token counting with specialized handling for Japanese text, considering character types, patterns, and language features
22. **Comprehensive Token Tracking**: Detailed monitoring of token usage by request, model, time period, with estimated savings calculation
23. **Dynamic Message Truncation**: Intelligently preserves the most important parts of messages when truncation is needed
24. **Session Analytics**: Provides insights into token usage patterns to help identify optimization opportunities

These optimizations work together to provide a responsive, efficient, and accurate experience while minimizing API token consumption and maximizing performance.

## File Structure

- `play_voice_with_gemini.py`: The main Python script
- `voiceData.csv`: CSV file mapping audio filenames to their text content
- `Voice/`: Directory containing WAV audio files
- `Log.txt`: Automatically generated file containing conversation logs
- `system_prompt.txt`: Optional file containing custom system prompt instructions
- `.env`: Configuration file for environment variables like the Gemini CLI path

## Troubleshooting

- If you encounter encoding issues with Japanese text, the script will try different encodings (utf-8, shift-jis, euc-jp, iso-2022-jp)
- Make sure the Gemini CLI is properly installed and configured on your system
- Ensure the path to the Gemini CLI in your .env file is correct
- Ensure all required packages are installed
- The script uses the 'gemini-2.5-flash' model with fallback to 'gemini-2.5-pro' and 'gemini-pro-vision' if needed
- If you encounter errors with the Gemini CLI:
  1. Check that you're authenticated with the CLI by running `gemini auth login`
  2. Verify that the models are available by running `gemini models list`
  3. Check for any error messages in the CLI output
- If you encounter quota limit errors, the script will:
  1. Immediately try the next available model without retrying (to save tokens)
  2. Provide clear error messages about quota limitations
  3. Use token-efficient modes (7 and 8) to minimize API usage
- If you're frequently hitting quota limits, consider:
  1. Spacing out your requests
  2. Upgrading to a paid API tier for higher quotas
