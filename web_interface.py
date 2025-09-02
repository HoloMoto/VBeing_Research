from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import subprocess
import pygame
import threading
import time
from pathlib import Path

# Import functionality from the original script
from play_voice_with_gemini import (
    run_gemini_command, 
    load_voice_data, 
    play_audio, 
    log_conversation,
    query_with_retries,
    get_best_matching_audio,
    process_gemini_response,
    initialize_models,
    generate_zonos_voice,
    get_available_clone_voices,
    get_quota_usage_stats,
    test_zonos_connection,
    load_system_prompt
)

# Import settings module
from settings import load_settings, save_settings, update_settings, get_setting

def play_error_sound():
    """
    Play the error sound when an error occurs.
    Uses the default error sound file at Voice/DefaultSystemVoice/error.wav.
    """
    error_sound_path = os.path.join('Voice', 'DefaultSystemVoice', 'error.wav')
    try:
        # Play the error sound in a separate thread to avoid blocking
        threading.Thread(target=play_audio, args=(error_sound_path,)).start()
        print("Playing error sound")
    except Exception as e:
        print(f"Failed to play error sound: {e}")

def is_error_response(response):
    """
    Check if a Gemini response contains an error.

    Args:
        response (str): The response from Gemini

    Returns:
        bool: True if the response contains an error, False otherwise
    """
    if not response:
        return True

    # Check for explicit error indicators
    error_indicators = [
        "Error:", 
        "not found", 
        "quota", 
        "rate limit",
        "エラー",
        "申し訳ありません",
        "I apologize",
        "I'm sorry",
        "I cannot",
        "I'm not able to",
        "I am unable to"
    ]

    for indicator in error_indicators:
        if indicator.lower() in response.lower():
            # Play error sound when Gemini returns an error response
            play_error_sound()
            return True

    return False

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Constants
SYSTEM_PROMPT_TEMPLATES_DIR = 'system_prompt_templates'

# Global variables
conversation_history = []
current_mode = None
voice_data = None
models = None
current_model = None  # Currently selected model
system_prompt = None  # System prompt to guide Gemini's responses
user_settings = None  # User settings loaded from settings.json

# Cache for Zonos connection status
zonos_connection_cache = {
    'status': None,
    'message': None,
    'timestamp': 0,
    'valid_for': 60  # Cache valid for 60 seconds
}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def initialize():
    """Initialize the application"""
    global voice_data, models, system_prompt, user_settings, current_mode, current_model

    try:
        # Load user settings
        user_settings = load_settings()
        print("User settings loaded:", user_settings)

        # Load voice data
        voice_data = load_voice_data()

        # Initialize models
        models = initialize_models()

        # Load system prompt
        system_prompt = load_system_prompt()
        if system_prompt:
            print("System prompt loaded successfully")
        else:
            print("Could not load system prompt. Using default.")

        # Get available models from CLI_AVAILABLE_MODELS
        available_models = []
        if hasattr(initialize_models, 'CLI_AVAILABLE_MODELS'):
            available_models = initialize_models.CLI_AVAILABLE_MODELS
        elif 'CLI_AVAILABLE_MODELS' in globals():
            available_models = globals()['CLI_AVAILABLE_MODELS']

        # If no models were found, use default models
        if not available_models:
            available_models = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-pro-vision']

        # Apply saved settings
        if user_settings:
            # Set current model from settings if it's available
            if "model" in user_settings and user_settings["model"] in available_models:
                current_model = user_settings["model"]
                print(f"Applied saved model setting: {current_model}")

            # Set current mode from settings
            if "mode" in user_settings and user_settings["mode"] in [1, 2, 5, 6, 7, 8, 9]:
                current_mode = user_settings["mode"]
                # Also load the appropriate system prompt for this mode
                mode_system_prompt = load_mode_specific_system_prompt(current_mode)
                if mode_system_prompt:
                    system_prompt = mode_system_prompt
                print(f"Applied saved mode setting: {current_mode}")

        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully',
            'modes': [
                {'id': 1, 'name': 'Direct mode'},
                {'id': 2, 'name': 'Text matching mode'},
                {'id': 5, 'name': 'Speech recognition mode'},
                {'id': 6, 'name': 'Text-only mode'},
                {'id': 7, 'name': 'RAG mode'},
                {'id': 8, 'name': 'Speech RAG mode'},
                {'id': 9, 'name': 'Zonos voice generation mode'}
            ],
            'available_models': [{'id': model, 'name': model} for model in available_models],
            'current_settings': {
                'model': current_model,
                'mode': current_mode,
                'voice': user_settings.get('voice', {}) if user_settings else {}
            }
        })
    except Exception as e:
        # Play error sound when initialization fails
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Initialization failed: {str(e)}'
        }), 500

def check_zonos_connection():
    """
    Check the Zonos connection status using the cache.
    If the cache is valid, return the cached status.
    Otherwise, perform a new connection test and update the cache.

    Returns:
        tuple: (success, message)
    """
    global zonos_connection_cache

    current_time = time.time()

    # Check if cache is valid
    if (zonos_connection_cache['status'] is not None and 
        current_time - zonos_connection_cache['timestamp'] < zonos_connection_cache['valid_for']):
        # Use cached status
        print("Using cached Zonos connection status")
        return zonos_connection_cache['status'], zonos_connection_cache['message']

    # Perform new connection test
    print("Performing new Zonos connection test")
    success, message = test_zonos_connection()

    # Update cache
    zonos_connection_cache['status'] = success
    zonos_connection_cache['message'] = message
    zonos_connection_cache['timestamp'] = current_time

    return success, message

def load_mode_specific_system_prompt(mode):
    """
    Load a mode-specific system prompt from the system_prompt_templates directory.

    Args:
        mode: The mode number (1, 2, 5, 6, 7, 8, or 9)

    Returns:
        The system prompt as a string, or None if the file could not be read
    """
    # Map mode numbers to file names
    mode_file_map = {
        1: "mode1_direct.txt",
        2: "mode2_text_matching.txt",
        5: "mode5_speech_recognition.txt",
        6: "mode6_text_only.txt",
        7: "mode7_rag.txt",
        8: "mode8_speech_rag.txt",
        9: "mode9_zonos.txt"
    }

    if mode not in mode_file_map:
        print(f"No system prompt template for mode {mode}")
        return load_system_prompt()  # Fall back to default system prompt

    template_file = os.path.join("system_prompt_templates", mode_file_map[mode])

    try:
        # Try different encodings since Japanese text might be encoded differently
        encodings = ['utf-8', 'shift-jis', 'euc-jp', 'iso-2022-jp']

        for encoding in encodings:
            try:
                with open(template_file, 'r', encoding=encoding) as file:
                    system_prompt = file.read()
                print(f"Successfully loaded system prompt for mode {mode} with encoding: {encoding}")
                return system_prompt
            except UnicodeDecodeError:
                continue  # Try the next encoding
            except Exception as e:
                print(f"Error with encoding {encoding}: {e}")
                play_error_sound()
                continue

        print(f"Could not read system prompt template for mode {mode} with any encoding. Using default system prompt.")
        return load_system_prompt()  # Fall back to default system prompt
    except Exception as e:
        print(f"Error loading system prompt template for mode {mode}: {e}")
        play_error_sound()
        return load_system_prompt()  # Fall back to default system prompt

@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """Set the current mode"""
    global current_mode, system_prompt, user_settings

    data = request.json
    mode = data.get('mode')

    if mode not in [1, 2, 5, 6, 7, 8, 9]:
        return jsonify({
            'status': 'error',
            'message': 'Invalid mode selected'
        }), 400

    # If Zonos mode is selected, test the connection first
    if mode == 9:
        success, message = check_zonos_connection()
        if not success:
            # Play error sound when Zonos connection fails
            play_error_sound()
            return jsonify({
                'status': 'warning',
                'message': f'Zonosモードを有効にしましたが、接続テストに失敗しました: {message}'
            })

    # Load mode-specific system prompt
    system_prompt = load_mode_specific_system_prompt(mode)
    if system_prompt:
        print(f"Loaded system prompt for mode {mode}")
    else:
        print(f"Could not load system prompt for mode {mode}. Using default.")

    current_mode = mode

    # Save the mode setting
    if user_settings is None:
        user_settings = load_settings()

    user_settings["mode"] = mode
    save_settings(user_settings)
    print(f"Saved mode setting: {mode}")

    return jsonify({
        'status': 'success',
        'message': f'Mode set to {mode}'
    })

@app.route('/api/set_model', methods=['POST'])
def set_model():
    """Set the current model"""
    global current_model, user_settings

    data = request.json
    model = data.get('model')

    # Get available models
    available_models = []
    if hasattr(initialize_models, 'CLI_AVAILABLE_MODELS'):
        available_models = initialize_models.CLI_AVAILABLE_MODELS
    elif 'CLI_AVAILABLE_MODELS' in globals():
        available_models = globals()['CLI_AVAILABLE_MODELS']

    # If no models were found, use default models
    if not available_models:
        available_models = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-pro-vision']

    # Check if the model is valid
    if model not in available_models:
        return jsonify({
            'status': 'error',
            'message': f'Invalid model selected: {model}'
        }), 400

    current_model = model

    # Save the model setting
    if user_settings is None:
        user_settings = load_settings()

    user_settings["model"] = model
    save_settings(user_settings)
    print(f"Saved model setting: {model}")

    return jsonify({
        'status': 'success',
        'message': f'Model set to {model}'
    })

@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Process a user message"""
    global conversation_history, current_mode, voice_data, models, current_model, system_prompt

    if current_mode is None:
        return jsonify({
            'status': 'error',
            'message': 'Please select a mode first'
        }), 400

    data = request.json
    user_input = data.get('message', '').strip()

    if not user_input:
        return jsonify({
            'status': 'error',
            'message': 'Empty message'
        }), 400

    # Add user message to conversation history
    conversation_history.append({"role": "user", "parts": [user_input]})

    try:
        # Process based on mode
        if current_mode == 6:  # Text-only mode
            # Query Gemini directly with the selected model if available
            gemini_response = query_with_retries(models, conversation_history, preferred_model=current_model, system_prompt=system_prompt)
            audio_file = None

        elif current_mode == 9:  # Zonos voice generation mode
            # Query Gemini with the selected model if available
            gemini_response = query_with_retries(models, conversation_history, preferred_model=current_model, system_prompt=system_prompt)

            # Check if the Gemini response contains an error
            if is_error_response(gemini_response):
                print(f"Error detected in Gemini response, skipping Zonos processing: {gemini_response[:100]}...")
                # Add Gemini's response to conversation history
                conversation_history.append({"role": "model", "parts": [gemini_response]})
                # Return the error response to the user without calling Zonos
                return jsonify({
                    'status': 'warning',
                    'response': gemini_response,
                    'audio_file': None,
                    'model_used': current_model or 'auto',
                    'message': 'Geminiの応答にエラーが検出されたため、音声生成をスキップしました。'
                })

            # Check if clone voice is specified in the request
            use_clone_voice = data.get('use_clone_voice', None)
            clone_voice_file = data.get('clone_voice_file', None)

            # If not specified in the request, use saved settings
            if use_clone_voice is None and user_settings and 'voice' in user_settings:
                use_clone_voice = user_settings['voice'].get('use_clone_voice', False)
                print(f"Using saved voice setting: use_clone_voice={use_clone_voice}")

            if clone_voice_file is None and user_settings and 'voice' in user_settings:
                clone_voice_file = user_settings['voice'].get('clone_voice_file', None)
                print(f"Using saved voice setting: clone_voice_file={clone_voice_file}")

            # Check Zonos connection using cache
            success, message = check_zonos_connection()
            if not success:
                # Play error sound when Zonos API connection fails
                play_error_sound()
                return jsonify({
                    'status': 'error',
                    'message': f'Zonos API接続エラー: {message}'
                }), 500

            # Set audio_file to None initially, will be updated asynchronously
            audio_file = None

            # Start asynchronous voice generation
            def generate_and_save_voice():
                nonlocal audio_file
                try:
                    # Generate voice using Zonos API
                    audio_data = None
                    filename = None

                    # Get the voice data from Zonos API
                    from play_voice_with_gemini import generate_zonos_voice_data, save_zonos_voice_data

                    # Generate the voice data (this is the slow part)
                    audio_data = generate_zonos_voice_data(
                        gemini_response, 
                        use_clone_voice=use_clone_voice, 
                        clone_voice_file=clone_voice_file,
                        system_prompt=system_prompt
                    )

                    if audio_data:
                        # Store the audio data in a global cache for streaming to the browser
                        # This will be accessed by the /api/stream_audio endpoint
                        audio_cache_key = f"audio_{int(time.time() * 1000)}"
                        if not hasattr(app, 'audio_cache'):
                            app.audio_cache = {}
                        app.audio_cache[audio_cache_key] = {
                            'data': audio_data,
                            'timestamp': time.time(),
                            'content_type': 'audio/webm'
                        }

                        # Set the audio_file to the cache key for the frontend to use
                        audio_file = audio_cache_key

                        # Save the audio data properly and update CSV in the background
                        # This happens asynchronously while the browser is already playing the audio
                        filename = save_zonos_voice_data(audio_data, gemini_response)

                        # Update the permanent filename in the cache for future reference
                        if audio_cache_key in app.audio_cache:
                            app.audio_cache[audio_cache_key]['permanent_filename'] = filename

                        # Clean up old cache entries (older than 10 minutes)
                        current_time = time.time()
                        keys_to_remove = [k for k, v in app.audio_cache.items() 
                                         if current_time - v['timestamp'] > 600]
                        for k in keys_to_remove:
                            del app.audio_cache[k]
                except Exception as e:
                    print(f"Error in asynchronous voice generation: {e}")

            # Start the voice generation in a separate thread
            threading.Thread(target=generate_and_save_voice).start()

        elif current_mode in [1, 2, 5, 7, 8]:  # Direct, Text matching, Speech, RAG, or Speech RAG mode
            # Query Gemini with the selected model if available
            gemini_response = query_with_retries(models, conversation_history, preferred_model=current_model, system_prompt=system_prompt)

            # Get audio file based on mode
            if current_mode == 1:  # Direct mode
                audio_file = process_gemini_response(gemini_response)
            else:  # Text matching, Speech, RAG, or Speech RAG mode
                audio_file = get_best_matching_audio(gemini_response, voice_data)

            # Play audio in a separate thread to not block the response
            if audio_file:
                threading.Thread(target=play_audio, args=(audio_file,)).start()

        # Add Gemini's response to conversation history
        conversation_history.append({"role": "model", "parts": [gemini_response]})

        # For Zonos mode, we need to handle the audio_file differently
        if current_mode == 9:
            # Log the conversation with a placeholder for the audio file
            log_conversation(current_mode, user_input, gemini_response, "generating")

            # Return response immediately, audio will be generated asynchronously
            return jsonify({
                'status': 'success',
                'response': gemini_response,
                'audio_file': 'generating',  # Placeholder value
                'model_used': current_model or 'auto',
                'async_audio': True,  # Flag to indicate async audio generation
                'stream_audio': True  # Flag to indicate that audio should be streamed from the API
            })
        else:
            # For other modes, log the conversation normally
            log_conversation(current_mode, user_input, gemini_response, audio_file)

            # Return response with audio file info
            return jsonify({
                'status': 'success',
                'response': gemini_response,
                'audio_file': audio_file,
                'model_used': current_model or 'auto'
            })

    except Exception as e:
        # Play error sound when message processing fails
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error processing message: {str(e)}'
        }), 500

@app.route('/api/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset the conversation history"""
    global conversation_history

    conversation_history = []

    return jsonify({
        'status': 'success',
        'message': 'Conversation reset'
    })

@app.route('/api/direct_tts', methods=['POST'])
def direct_tts():
    """Generate voice directly using Zonos API for debugging purposes"""
    try:
        global user_settings

        data = request.json
        text = data.get('text', '').strip()

        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Empty text'
            }), 400

        # Check if clone voice is specified in the request
        use_clone_voice = data.get('use_clone_voice', None)
        clone_voice_file = data.get('clone_voice_file', None)

        # If not specified in the request, use saved settings
        if use_clone_voice is None and user_settings and 'voice' in user_settings:
            use_clone_voice = user_settings['voice'].get('use_clone_voice', False)
            print(f"Using saved voice setting: use_clone_voice={use_clone_voice}")

        if clone_voice_file is None and user_settings and 'voice' in user_settings:
            clone_voice_file = user_settings['voice'].get('clone_voice_file', None)
            print(f"Using saved voice setting: clone_voice_file={clone_voice_file}")

        # Check Zonos connection using cache
        success, message = check_zonos_connection()
        if not success:
            # Play error sound when Zonos API connection fails
            play_error_sound()
            return jsonify({
                'status': 'error',
                'message': f'Zonos API接続エラー: {message}'
            }), 500

        # Set audio_file to None initially, will be updated asynchronously
        audio_file = None

        # Start asynchronous voice generation
        def generate_and_save_voice():
            nonlocal audio_file
            try:
                # Generate voice using Zonos API
                audio_data = None
                filename = None

                # Get the voice data from Zonos API
                from play_voice_with_gemini import generate_zonos_voice_data, save_zonos_voice_data

                # Generate the voice data (this is the slow part)
                audio_data = generate_zonos_voice_data(
                    text, 
                    use_clone_voice=use_clone_voice, 
                    clone_voice_file=clone_voice_file,
                    system_prompt=system_prompt
                )

                if audio_data:
                    # Store the audio data in a global cache for streaming to the browser
                    # This will be accessed by the /api/stream_audio endpoint
                    audio_cache_key = f"audio_{int(time.time() * 1000)}"
                    if not hasattr(app, 'audio_cache'):
                        app.audio_cache = {}
                    app.audio_cache[audio_cache_key] = {
                        'data': audio_data,
                        'timestamp': time.time(),
                        'content_type': 'audio/webm'
                    }

                    # Set the audio_file to the cache key for the frontend to use
                    audio_file = audio_cache_key

                    # Save the audio data properly and update CSV in the background
                    # This happens asynchronously while the browser is already playing the audio
                    filename = save_zonos_voice_data(audio_data, text)

                    # Update the permanent filename in the cache for future reference
                    if audio_cache_key in app.audio_cache:
                        app.audio_cache[audio_cache_key]['permanent_filename'] = filename

                    # Clean up old cache entries (older than 10 minutes)
                    current_time = time.time()
                    keys_to_remove = [k for k, v in app.audio_cache.items() 
                                     if current_time - v['timestamp'] > 600]
                    for k in keys_to_remove:
                        del app.audio_cache[k]
            except Exception as e:
                print(f"Error in asynchronous voice generation: {e}")
                play_error_sound()

        # Start the voice generation in a separate thread
        threading.Thread(target=generate_and_save_voice).start()

        # Return a response immediately
        return jsonify({
            'status': 'success',
            'message': '音声生成を開始しました。生成が完了すると自動的に再生されます。',
            'audio_file': 'generating',  # Placeholder value
            'stream_audio': True  # Flag to indicate that audio should be streamed from the API
        })

    except Exception as e:
        # Play error sound when voice generation fails
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error generating voice: {str(e)}'
        }), 500

@app.route('/api/clone_voices', methods=['GET'])
def get_clone_voices():
    """Get available clone voices"""
    try:
        # Get available clone voices
        voices = get_available_clone_voices()

        return jsonify({
            'status': 'success',
            'voices': voices
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error getting clone voices: {str(e)}'
        }), 500

@app.route('/api/quota_usage', methods=['GET'])
def get_quota_usage():
    """Get quota usage statistics"""
    try:
        # Get quota usage statistics
        quota_stats = get_quota_usage_stats()

        return jsonify({
            'status': 'success',
            'quota_stats': quota_stats
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error getting quota usage statistics: {str(e)}'
        }), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    audio_dir = Path('Voice')
    file_path = audio_dir / filename

    if not file_path.exists():
        # Play error sound when audio file is not found
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Audio file {filename} not found'
        }), 404

    return send_from_directory(audio_dir, filename)

@app.route('/api/stream_audio/<cache_key>')
def stream_audio(cache_key):
    """Stream audio data directly from memory cache"""
    if not hasattr(app, 'audio_cache') or cache_key not in app.audio_cache:
        # Play error sound when audio data is not found in cache
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Audio data for key {cache_key} not found in cache'
        }), 404

    # Get the audio data from the cache
    audio_entry = app.audio_cache[cache_key]
    audio_data = audio_entry['data']
    content_type = audio_entry.get('content_type', 'audio/webm')

    # Return the audio data with the appropriate content type
    response = app.response_class(
        response=audio_data,
        status=200,
        mimetype=content_type
    )

    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response

@app.route('/api/check_audio_status')
def check_audio_status():
    """Check if audio data is available in the cache"""
    if not hasattr(app, 'audio_cache') or not app.audio_cache:
        return jsonify({
            'status': 'pending',
            'message': 'No audio data available yet'
        })

    # Find the most recent audio entry
    most_recent_key = None
    most_recent_time = 0

    for key, entry in app.audio_cache.items():
        if entry['timestamp'] > most_recent_time:
            most_recent_time = entry['timestamp']
            most_recent_key = key

    if most_recent_key:
        # Check if the entry is recent (within the last 10 seconds)
        if time.time() - most_recent_time < 10:
            return jsonify({
                'status': 'success',
                'audio_key': most_recent_key,
                'timestamp': most_recent_time
            })

    # No recent audio data found
    return jsonify({
        'status': 'pending',
        'message': 'No recent audio data available'
    })

@app.route('/api/convert_audio', methods=['POST'])
def convert_audio():
    """Trigger batch conversion of webm files to wav format"""
    try:
        from play_voice_with_gemini import batch_convert_webm_to_wav

        # Start the conversion in a background thread to avoid blocking the response
        threading.Thread(target=batch_convert_webm_to_wav).start()

        return jsonify({
            'status': 'success',
            'message': 'Batch conversion started in the background'
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error starting batch conversion: {str(e)}'
        }), 500

@app.route('/api/check_model_file', methods=['GET'])
def check_model_file():
    """Check if a model file exists on the server"""
    try:
        model_path = request.args.get('path', '')
        if not model_path:
            return jsonify({
                'status': 'error',
                'message': 'No model path provided'
            }), 400

        # Remove leading slash if present
        if model_path.startswith('/'):
            model_path = model_path[1:]

        # Check if the file exists
        full_path = os.path.join(os.getcwd(), model_path)
        exists = os.path.isfile(full_path)

        # Get file info if it exists
        file_info = {}
        if exists:
            file_info = {
                'size': os.path.getsize(full_path),
                'modified': os.path.getmtime(full_path),
                'path': full_path
            }

        return jsonify({
            'status': 'success',
            'exists': exists,
            'file_info': file_info
        })
    except Exception as e:
        print(f"Error checking model file: {str(e)}")
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error checking model file: {str(e)}'
        }), 500

@app.route('/api/log_character_event', methods=['POST'])
def log_character_event():
    """Log character initialization events and errors to the server"""
    try:
        data = request.json
        event_type = data.get('event_type', 'unknown')
        message = data.get('message', '')
        details = data.get('details', {})

        # Log the event to the server log
        print(f"[Character Event] {event_type}: {message}")
        if details:
            print(f"[Character Event Details] {json.dumps(details, indent=2)}")

        # Log to the Log.txt file
        with open('Log.txt', 'a', encoding='utf-8') as log_file:
            timestamp = time.strftime('[%Y-%m-%d %H:%M:%S]')
            log_file.write(f"{timestamp} [Character {event_type}] {message}\n")
            if details:
                log_file.write(f"{timestamp} [Character Details] {json.dumps(details, ensure_ascii=False)}\n")

        return jsonify({
            'status': 'success',
            'message': 'Event logged successfully'
        })
    except Exception as e:
        print(f"Error logging character event: {str(e)}")
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error logging event: {str(e)}'
        }), 500

# System Prompt API Endpoints
@app.route('/api/system_prompt/current', methods=['GET'])
def get_current_system_prompt():
    """Get the current system prompt"""
    global system_prompt

    try:
        # Use the global system_prompt variable if it exists
        if system_prompt is None:
            # If not, load it from the file
            system_prompt = load_system_prompt()

        if system_prompt is None:
            return jsonify({
                'status': 'error',
                'message': 'System prompt file not found'
            }), 404

        return jsonify({
            'status': 'success',
            'content': system_prompt
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error loading system prompt: {str(e)}'
        }), 500

@app.route('/api/system_prompt/current', methods=['POST'])
def save_current_system_prompt():
    """Save the current system prompt"""
    global system_prompt

    try:
        data = request.json
        content = data.get('content', '').strip()

        if not content:
            return jsonify({
                'status': 'error',
                'message': 'Empty system prompt'
            }), 400

        # Get the system prompt path from environment variable or use default
        system_prompt_path = os.getenv('SYSTEM_PROMPT_PATH', 'system_prompt.txt')

        # Save the system prompt
        with open(system_prompt_path, 'w', encoding='utf-8') as file:
            file.write(content)

        # Update the global system_prompt variable
        system_prompt = content
        print("System prompt updated successfully")

        return jsonify({
            'status': 'success',
            'message': 'System prompt saved successfully'
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error saving system prompt: {str(e)}'
        }), 500

@app.route('/api/system_prompt/templates', methods=['GET'])
def get_system_prompt_templates():
    """Get all system prompt templates"""
    try:
        # Create templates directory if it doesn't exist
        os.makedirs(SYSTEM_PROMPT_TEMPLATES_DIR, exist_ok=True)

        templates = []

        # Get all template files
        for filename in os.listdir(SYSTEM_PROMPT_TEMPLATES_DIR):
            if filename.endswith('.txt'):
                # Extract template name from filename
                name = os.path.splitext(filename)[0]

                templates.append({
                    'name': name,
                    'filename': filename
                })

        return jsonify({
            'status': 'success',
            'templates': templates
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error getting templates: {str(e)}'
        }), 500

@app.route('/api/system_prompt/templates/<filename>', methods=['GET'])
def get_system_prompt_template(filename):
    """Get a specific system prompt template"""
    try:
        # Validate filename
        if not filename.endswith('.txt'):
            filename += '.txt'

        # Check if file exists
        template_path = os.path.join(SYSTEM_PROMPT_TEMPLATES_DIR, filename)
        if not os.path.exists(template_path):
            return jsonify({
                'status': 'error',
                'message': f'Template {filename} not found'
            }), 404

        # Read template content
        with open(template_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Extract template name from filename
        name = os.path.splitext(filename)[0]

        return jsonify({
            'status': 'success',
            'name': name,
            'content': content
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error getting template: {str(e)}'
        }), 500

@app.route('/api/system_prompt/templates', methods=['POST'])
def save_system_prompt_template():
    """Save a new system prompt template"""
    try:
        data = request.json
        name = data.get('name', '').strip()
        content = data.get('content', '').strip()

        if not name:
            return jsonify({
                'status': 'error',
                'message': 'Empty template name'
            }), 400

        if not content:
            return jsonify({
                'status': 'error',
                'message': 'Empty template content'
            }), 400

        # Create templates directory if it doesn't exist
        os.makedirs(SYSTEM_PROMPT_TEMPLATES_DIR, exist_ok=True)

        # Sanitize filename
        filename = f"{name.replace(' ', '_')}.txt"

        # Save template
        template_path = os.path.join(SYSTEM_PROMPT_TEMPLATES_DIR, filename)
        with open(template_path, 'w', encoding='utf-8') as file:
            file.write(content)

        return jsonify({
            'status': 'success',
            'message': f'Template {name} saved successfully',
            'filename': filename
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error saving template: {str(e)}'
        }), 500

@app.route('/api/system_prompt/templates/<filename>', methods=['DELETE'])
def delete_system_prompt_template(filename):
    """Delete a system prompt template"""
    try:
        # Validate filename
        if not filename.endswith('.txt'):
            filename += '.txt'

        # Check if file exists
        template_path = os.path.join(SYSTEM_PROMPT_TEMPLATES_DIR, filename)
        if not os.path.exists(template_path):
            return jsonify({
                'status': 'error',
                'message': f'Template {filename} not found'
            }), 404

        # Delete template
        os.remove(template_path)

        return jsonify({
            'status': 'success',
            'message': f'Template {filename} deleted successfully'
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error deleting template: {str(e)}'
        }), 500

# Settings API Endpoints
@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get the current settings"""
    global user_settings

    try:
        # If settings haven't been loaded yet, load them
        if user_settings is None:
            user_settings = load_settings()

        return jsonify({
            'status': 'success',
            'settings': user_settings
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error getting settings: {str(e)}'
        }), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    global user_settings

    try:
        data = request.json
        settings_to_update = data.get('settings', {})

        # If settings haven't been loaded yet, load them
        if user_settings is None:
            user_settings = load_settings()

        # Update settings
        for key, value in settings_to_update.items():
            user_settings[key] = value

        # Save updated settings
        save_settings(user_settings)
        print(f"Settings updated: {settings_to_update}")

        return jsonify({
            'status': 'success',
            'message': 'Settings updated successfully',
            'settings': user_settings
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error updating settings: {str(e)}'
        }), 500

@app.route('/api/settings/voice', methods=['POST'])
def update_voice_settings():
    """Update voice settings"""
    global user_settings

    try:
        data = request.json
        use_clone_voice = data.get('use_clone_voice', False)
        clone_voice_file = data.get('clone_voice_file', None)

        # If settings haven't been loaded yet, load them
        if user_settings is None:
            user_settings = load_settings()

        # Update voice settings
        if 'voice' not in user_settings:
            user_settings['voice'] = {}

        user_settings['voice']['use_clone_voice'] = use_clone_voice
        user_settings['voice']['clone_voice_file'] = clone_voice_file

        # Save updated settings
        save_settings(user_settings)
        print(f"Voice settings updated: use_clone_voice={use_clone_voice}, clone_voice_file={clone_voice_file}")

        return jsonify({
            'status': 'success',
            'message': 'Voice settings updated successfully',
            'voice_settings': user_settings['voice']
        })
    except Exception as e:
        play_error_sound()
        return jsonify({
            'status': 'error',
            'message': f'Error updating voice settings: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Create system prompt templates directory if it doesn't exist
    os.makedirs(SYSTEM_PROMPT_TEMPLATES_DIR, exist_ok=True)

    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
