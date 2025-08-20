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
    play_audio_file, 
    log_conversation,
    query_with_retries,
    get_best_matching_audio,
    process_gemini_response,
    initialize_models
)

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Global variables
conversation_history = []
current_mode = None
voice_data = None
models = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def initialize():
    """Initialize the application"""
    global voice_data, models

    try:
        # Load voice data
        voice_data = load_voice_data()

        # Initialize models
        models = initialize_models()

        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully',
            'modes': [
                {'id': 1, 'name': 'Direct mode'},
                {'id': 2, 'name': 'Text matching mode'},
                {'id': 5, 'name': 'Speech recognition mode'},
                {'id': 6, 'name': 'Text-only mode'},
                {'id': 7, 'name': 'RAG mode'},
                {'id': 8, 'name': 'Speech RAG mode'}
            ]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Initialization failed: {str(e)}'
        }), 500

@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """Set the current mode"""
    global current_mode

    data = request.json
    mode = data.get('mode')

    if mode not in [1, 2, 5, 6, 7, 8]:
        return jsonify({
            'status': 'error',
            'message': 'Invalid mode selected'
        }), 400

    current_mode = mode

    return jsonify({
        'status': 'success',
        'message': f'Mode set to {mode}'
    })

@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Process a user message"""
    global conversation_history, current_mode, voice_data, models

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
            # Query Gemini directly
            gemini_response = query_with_retries(models, conversation_history)
            audio_file = None

        elif current_mode in [1, 2, 5, 7, 8]:  # Direct, Text matching, Speech, RAG, or Speech RAG mode
            # Query Gemini
            gemini_response = query_with_retries(models, conversation_history)

            # Get audio file based on mode
            if current_mode == 1:  # Direct mode
                audio_file = process_gemini_response(gemini_response)
            else:  # Text matching, Speech, RAG, or Speech RAG mode
                audio_file = get_best_matching_audio(gemini_response, voice_data)

            # Play audio in a separate thread to not block the response
            if audio_file:
                threading.Thread(target=play_audio_file, args=(audio_file,)).start()

        # Add Gemini's response to conversation history
        conversation_history.append({"role": "model", "parts": [gemini_response]})

        # Log the conversation
        log_conversation(current_mode, user_input, gemini_response, audio_file)

        return jsonify({
            'status': 'success',
            'response': gemini_response,
            'audio_file': audio_file
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

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
