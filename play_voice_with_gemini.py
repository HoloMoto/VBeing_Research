"""
Voice Playback with Gemini

This script uses Google's Gemini AI to play audio files from the Voice/ directory
according to the data in voiceData.csv. It provides five modes of operation:

1. Direct mode: Gemini selects the audio file directly
2. Text matching mode: Match Gemini's response with voice data text
3. Manual selection mode: Select an audio file by number
4. List available models: Display all available Gemini models (for troubleshooting)
5. Speech recognition mode: Use microphone input instead of typing

Requirements:
- Python 3.7+
- pygame library (for audio playback)
- google-generativeai library (for Gemini API)
- speech_recognition library (for speech recognition)
- Access to one of these Gemini models: 'gemini-1.5-flash', 'gemini-1.5-pro', or 'gemini-pro'

Setup:
1. An API key is already configured in the script. If you need to use your own:
   - Get a Gemini API key from https://ai.google.dev/
   - Replace the existing API key with your own
2. Install required packages: pip install pygame google-generativeai SpeechRecognition pyaudio

Usage:
- Run the script: python play_voice_with_gemini.py
- Select a mode (1, 2, 3, 4, or 5) once at the beginning
- Continue the conversation with Gemini in a continuous chat session
- Type or say 'quit' to exit or 'change mode' to select a different mode

Performance Optimizations:
- The script now maintains conversation history for more contextual responses
- Model instances are created once and reused for faster response times
- Continuous chat mode eliminates the need to select a mode for each interaction

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
import google.generativeai as genai
import speech_recognition as sr
from pathlib import Path

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Configure the Gemini API with your API key
# You can get an API key from https://ai.google.dev/
GOOGLE_API_KEY = "AIzaSyAFdis6AZLJHFpR9dJKdbClZwlpV-HuJ8s"  # Current API key
genai.configure(api_key=GOOGLE_API_KEY)

# Path to the voice data CSV and voice directory
VOICE_DATA_CSV = "voiceData.csv"
VOICE_DIR = "Voice"

def load_voice_data():
    """Load the voice data from the CSV file."""
    voice_data = {}
    try:
        # Try different encodings since Japanese text might be encoded differently
        encodings = ['utf-8', 'shift-jis', 'euc-jp', 'iso-2022-jp']

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
                                voice_data[filename] = text
                print(f"Successfully loaded voice data with encoding: {encoding}")
                break  # If we get here, we've successfully read the file
            except UnicodeDecodeError:
                continue  # Try the next encoding
            except Exception as e:
                print(f"Error with encoding {encoding}: {e}")
                continue
    except Exception as e:
        print(f"Error loading voice data: {e}")

    return voice_data

def play_audio(filename):
    """Play the specified audio file."""
    try:
        file_path = os.path.join(VOICE_DIR, filename)
        if os.path.exists(file_path):
            print(f"Playing: {filename}")
            sound = pygame.mixer.Sound(file_path)
            sound.play()
            # Wait for the audio to finish playing
            pygame.time.wait(int(sound.get_length() * 1000))
        else:
            print(f"Audio file not found: {file_path}")
    except Exception as e:
        print(f"Error playing audio: {e}")

def list_available_models():
    """
    List all available models from the Gemini API.
    This is useful for troubleshooting when models are not found.

    Returns:
        A list of available model names
    """
    try:
        available_models = []
        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                available_models.append(model.name)
                print(f"Available model: {model.name}")
        return available_models
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

# Create model instances once at the beginning
MODEL_INSTANCES = {}
CHAT_SESSIONS = {}

def initialize_models():
    """Initialize model instances for reuse."""
    global MODEL_INSTANCES
    models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']

    for model_name in models:
        try:
            print(f"Initializing model: {model_name}")
            MODEL_INSTANCES[model_name] = genai.GenerativeModel(model_name)
        except Exception as e:
            print(f"Could not initialize model {model_name}: {e}")

def get_gemini_response(prompt, system_prompt=None, chat_history=None, max_retries=1, initial_retry_delay=1):
    """
    Get a response from Gemini based on the prompt.

    Args:
        prompt: The user prompt to send to Gemini
        system_prompt: Optional system prompt to guide Gemini's response
        chat_history: Optional chat history for maintaining conversation context
        max_retries: Maximum number of retry attempts for rate limit errors (default: 1)
        initial_retry_delay: Initial delay in seconds before retrying (default: 1)

    Returns:
        The text response from Gemini or None if an error occurred
    """
    global MODEL_INSTANCES, CHAT_SESSIONS
    models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']  # Fallback models in order of preference

    # Initialize models if not already done
    if not MODEL_INSTANCES:
        initialize_models()

    for model_name in models:
        # Skip if model wasn't successfully initialized
        if model_name not in MODEL_INSTANCES:
            continue

        retry_count = 0
        retry_delay = initial_retry_delay
        model = MODEL_INSTANCES[model_name]

        # Create a new chat session if needed
        if chat_history is not None and model_name not in CHAT_SESSIONS:
            try:
                CHAT_SESSIONS[model_name] = model.start_chat(history=[])
            except Exception as e:
                print(f"Could not start chat session for {model_name}: {e}")
                continue

        while retry_count <= max_retries:
            try:
                print(f"Using model: {model_name}")

                # Use chat session if available
                if chat_history is not None and model_name in CHAT_SESSIONS:
                    chat = CHAT_SESSIONS[model_name]

                    # Add system prompt if provided and not already in history
                    if system_prompt and not chat_history:
                        chat.send_message(system_prompt, role="system")

                    response = chat.send_message(prompt)
                    return response.text
                else:
                    # One-off generation without chat history
                    if system_prompt:
                        response = model.generate_content([
                            {"role": "system", "parts": [system_prompt]},
                            {"role": "user", "parts": [prompt]}
                        ])
                    else:
                        response = model.generate_content(prompt)

                    return response.text

            except Exception as e:
                error_str = str(e)

                # Check if it's a quota limit error
                if "429" in error_str and "quota" in error_str.lower():
                    if retry_count < max_retries:
                        print(f"Quota limit reached. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_count += 1
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"Quota limit reached for model {model_name}. Trying next model if available.")
                        break  # Try the next model
                else:
                    # For other errors, check if it's a model not found error
                    if "404" in error_str and "not found" in error_str.lower():
                        print(f"Model '{model_name}' not found. Listing available models...")
                        list_available_models()
                        print(f"Trying next model if available.")
                        break  # Try the next model
                    else:
                        # For other errors, just print and return None
                        print(f"Error getting Gemini response: {e}")
                        return None

    print("Failed to get a response from any of the configured models.")
    print("This could be due to quota limits, unavailable models, or other API issues.")
    print("You can try again later or check the available models with list_available_models().")
    return None

def get_audio_selection_from_gemini(prompt, available_files, chat_history=True):
    """Ask Gemini to select an audio file based on the prompt and conversation context."""
    system_prompt = f"""
    You are an assistant that helps select appropriate audio responses for a natural conversation flow.
    Available audio files: {', '.join(available_files)}

    When the user asks a question or makes a statement, your task is to select the most appropriate
    audio file from the list above that would be the best response, considering the entire conversation
    context so far. The goal is to create a coherent, natural-sounding conversation.

    Important guidelines:
    1. Consider the conversation history and context when selecting a response
    2. Choose responses that make sense as a reply to the user's current input
    3. Maintain conversation coherence by selecting responses that follow logically from previous exchanges
    4. Avoid selecting the same audio file repeatedly unless it's specifically appropriate
    5. Consider the tone and intent of the user's message when selecting a response

    Respond ONLY with the filename (e.g., 'audio5.wav') of the most appropriate audio file.
    Do not include any other text in your response.
    """

    response = get_gemini_response(prompt, system_prompt, chat_history=chat_history)

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

def listen_for_speech(language="ja-JP"):
    """
    Capture audio from the microphone and convert it to text using speech recognition.

    Args:
        language: The language code for speech recognition (default: Japanese)

    Returns:
        The recognized text as a string, or None if recognition failed
    """
    recognizer = sr.Recognizer()

    print("Listening... (Speak now)")

    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)

            # Listen for audio input
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            print("Processing speech...")

            # Use Google Speech Recognition to convert audio to text
            text = recognizer.recognize_google(audio, language=language)

            print(f"Recognized: {text}")
            return text

    except sr.WaitTimeoutError:
        print("No speech detected within timeout period.")
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    except Exception as e:
        print(f"Error during speech recognition: {e}")

    return None

def main():
    """Main function to run the voice playback with Gemini."""
    print("Loading voice data...")
    voice_data = load_voice_data()

    if not voice_data:
        print("No voice data found. Exiting.")
        return

    print(f"Loaded {len(voice_data)} voice entries.")

    # Get list of available audio files
    available_files = list(voice_data.keys())

    # Initialize models at startup for faster responses
    print("Initializing Gemini models...")
    initialize_models()

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

    mode = None

    while True:
        if not mode:
            print("\n===========================")
            mode = input("Select mode (1, 2, 3, 4, 5) or 'quit' to exit: ")

            if mode.lower() == 'quit':
                break

            if mode not in ['1', '2', '3', '4', '5']:
                print("Invalid mode selection. Please try again.")
                mode = None
                continue

        if mode == '1':
            # Direct selection mode - continuous chat
            print("\n===== Direct Mode =====")
            print("Type 'quit' to exit or 'change mode' to select a different mode")

            # Reset conversation history when entering this mode
            mode_previous_selections = []

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
                selected_file = get_audio_selection_from_gemini(user_input, available_files, chat_history=True)

                if selected_file:
                    print(f"Gemini selected: {selected_file}")
                    play_audio(selected_file)

                    # Add selected file to history
                    mode_previous_selections.append(selected_file)
                else:
                    print("Gemini couldn't select an appropriate audio file.")

        elif mode == '2':
            # Text matching mode - continuous chat
            print("\n===== Text Matching Mode =====")
            print("Type 'quit' to exit or 'change mode' to select a different mode")

            # Reset conversation history when entering this mode
            mode_previous_responses = []
            mode_previous_selections = []

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

                response = get_gemini_response(user_input, chat_history=True)

                if response:
                    print(f"Gemini: {response}")

                    # Add current response to conversation history
                    mode_previous_responses.append(response)

                    # Find best match considering conversation context
                    best_match = find_best_match_text(
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
                    else:
                        print("No matching audio found.")
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

            # Reset conversation history when entering this mode
            mode_previous_responses = []
            mode_previous_selections = []

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
                response = get_gemini_response(user_input, chat_history=True)

                if response:
                    print(f"Gemini: {response}")

                    # Add current response to conversation history
                    mode_previous_responses.append(response)

                    # Find best match considering conversation context
                    best_match = find_best_match_text(
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
                    else:
                        print("No matching audio found.")
                else:
                    print("Failed to get a response from Gemini.")

if __name__ == "__main__":
    try:
        main()
    finally:
        pygame.mixer.quit()
        print("Program ended.")
