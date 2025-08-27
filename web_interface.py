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
    global voice_data, models, system_prompt

    try:
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
            'available_models': [{'id': model, 'name': model} for model in available_models]
        })
    except Exception as e:
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
                continue

        print(f"Could not read system prompt template for mode {mode} with any encoding. Using default system prompt.")
        return load_system_prompt()  # Fall back to default system prompt
    except Exception as e:
        print(f"Error loading system prompt template for mode {mode}: {e}")
        return load_system_prompt()  # Fall back to default system prompt

@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """Set the current mode"""
    global current_mode, system_prompt

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

    return jsonify({
        'status': 'success',
        'message': f'Mode set to {mode}'
    })

@app.route('/api/set_model', methods=['POST'])
def set_model():
    """Set the current model"""
    global current_model

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

            # Check if clone voice is specified
            use_clone_voice = data.get('use_clone_voice', False)
            clone_voice_file = data.get('clone_voice_file', None)

            # Check Zonos connection using cache
            success, message = check_zonos_connection()
            if not success:
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
        data = request.json
        text = data.get('text', '').strip()

        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Empty text'
            }), 400

        # Check if clone voice is specified
        use_clone_voice = data.get('use_clone_voice', False)
        clone_voice_file = data.get('clone_voice_file', None)

        # Check Zonos connection using cache
        success, message = check_zonos_connection()
        if not success:
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
        return jsonify({
            'status': 'error',
            'message': f'Audio file {filename} not found'
        }), 404

    return send_from_directory(audio_dir, filename)

@app.route('/api/stream_audio/<cache_key>')
def stream_audio(cache_key):
    """Stream audio data directly from memory cache"""
    if not hasattr(app, 'audio_cache') or cache_key not in app.audio_cache:
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
        return jsonify({
            'status': 'error',
            'message': f'Error starting batch conversion: {str(e)}'
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
        return jsonify({
            'status': 'error',
            'message': f'Error deleting template: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Create system prompt templates directory if it doesn't exist
    os.makedirs(SYSTEM_PROMPT_TEMPLATES_DIR, exist_ok=True)

    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
