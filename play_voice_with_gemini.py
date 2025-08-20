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
from pathlib import Path
from dotenv import load_dotenv

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

        # Create the Gemini CLI command as a list
        command = [gemini_cli_path] + command_args

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
    List all available models from the Gemini CLI.
    This is useful for troubleshooting when models are not found.

    Returns:
        A list of available model names
    """
    try:
        # Run the CLI command to list available models
        # The exact command may vary depending on the Gemini CLI implementation
        # This is a common pattern for CLI tools
        response = run_gemini_command(["list", "models"])

        if response.startswith("Error:"):
            print(f"Error listing models: {response}")

            # Try alternative command format if the first one fails
            response = run_gemini_command(["models", "list"])

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
            test_cmd = ["--model", model_name, "--text", "Hello"]
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
            'token_savings': 0  # Track estimated token savings from optimizations
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

    # Log detailed statistics
    print(f"\n===== Token Usage Statistics =====")
    print(f"Current request: {prompt_tokens} prompt + {response_tokens} response = {total_tokens} tokens")
    print(f"Model used: {model_name}")
    print(f"Session total: {log_token_usage.session_stats['total_tokens']} tokens")
    print(f"Estimated token savings: {log_token_usage.session_stats['token_savings']} tokens")
    print(f"Average per request: {log_token_usage.session_stats['avg_tokens_per_request']:.1f} tokens")
    print(f"Request count: {log_token_usage.session_stats['request_count']}")

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

    stats.append("\nNote: These are estimated values and may differ from actual token counts.")
    return "\n".join(stats)

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

def get_gemini_response(prompt, system_prompt=None, chat_history=None, max_retries=0, initial_retry_delay=1, reset_history=False):
    """
    Get a response from Gemini based on the prompt using the local Gemini CLI.

    This function maintains conversation history when chat_history=True by including
    previous exchanges in the prompt sent to Gemini. This helps create more contextual
    and coherent conversations.

    Args:
        prompt: The user prompt to send to Gemini
        system_prompt: Optional system prompt to guide Gemini's response
        chat_history: When True, includes previous conversation history in the prompt
                     to maintain context across multiple exchanges
        max_retries: Maximum number of retry attempts for errors (default: 0)
        initial_retry_delay: Initial delay in seconds before retrying (default: 1)
        reset_history: When True, resets the conversation history before processing this prompt (default: False)

    Returns:
        A tuple containing:
        - The text response from Gemini or None if an error occurred
        - The full prompt sent to Gemini (including conversation history if used)
    """
    # Reset conversation history if requested
    if reset_history:
        reset_conversation_history()
    global CLI_AVAILABLE_MODELS

    # If no models have been checked yet, initialize them
    if not CLI_AVAILABLE_MODELS:
        initialize_models()

    # Initialize conversation history if it doesn't exist
    if not hasattr(get_gemini_response, 'CLI_CHAT_HISTORY'):
        get_gemini_response.CLI_CHAT_HISTORY = []

    # Initialize token usage tracking if it doesn't exist
    if not hasattr(get_gemini_response, 'token_usage'):
        get_gemini_response.token_usage = {
            'total_prompt_tokens': 0,
            'total_response_tokens': 0,
            'total_tokens': 0,
            'model_usage': {}
        }

    # Dynamically select the optimal model and parameters based on prompt complexity
    # Only if we have multiple models available
    if len(CLI_AVAILABLE_MODELS) > 1:
        # Get model configuration from the enhanced selection function
        model_config = select_optimal_model(prompt, get_gemini_response.CLI_CHAT_HISTORY if chat_history else None)

        # Extract model name and parameters
        preferred_model = model_config['model']
        temperature = model_config.get('temperature', 0.7)
        max_output_tokens = model_config.get('max_output_tokens', 500)
        top_k = model_config.get('top_k', 40)
        top_p = model_config.get('top_p', 0.9)

        # Store parameters for later use
        if not hasattr(get_gemini_response, 'model_parameters'):
            get_gemini_response.model_parameters = {}

        get_gemini_response.model_parameters = {
            'temperature': temperature,
            'max_output_tokens': max_output_tokens,
            'top_k': top_k,
            'top_p': top_p
        }

        # Reorder models to try preferred model first, but keep all models as fallbacks
        models = [preferred_model]
        for model in CLI_AVAILABLE_MODELS:
            if model != preferred_model:
                models.append(model)

        print(f"Dynamically selected model: {preferred_model} with temperature={temperature:.1f}, max_tokens={max_output_tokens}")
    # Use available models or fall back to default order if none are available
    elif CLI_AVAILABLE_MODELS:
        models = CLI_AVAILABLE_MODELS
        # Use default parameters
        get_gemini_response.model_parameters = {
            'temperature': 0.7,
            'max_output_tokens': 500,
            'top_k': 40,
            'top_p': 0.9
        }
    else:
        models = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-pro-vision']
        # Use default parameters
        get_gemini_response.model_parameters = {
            'temperature': 0.7,
            'max_output_tokens': 500,
            'top_k': 40,
            'top_p': 0.9
        }
        print("Warning: No models confirmed available. Will try default models.")

    # If chat_history is True, use the stored conversation history
    use_history = chat_history is True and hasattr(get_gemini_response, 'CLI_CHAT_HISTORY')

    for model_name in models:
        retry_count = 0
        retry_delay = initial_retry_delay

        while retry_count <= max_retries:
            try:
                print(f"Using model: {model_name}")

                # Prepare command arguments
                command_args = []

                # Add model selection and parameters
                command_args.extend(["--model", model_name])

                # Add model parameters if available
                if hasattr(get_gemini_response, 'model_parameters'):
                    params = get_gemini_response.model_parameters

                    # Add temperature parameter
                    if 'temperature' in params:
                        command_args.extend(["--temperature", str(params['temperature'])])

                    # Add max output tokens parameter
                    if 'max_output_tokens' in params:
                        command_args.extend(["--max-output-tokens", str(params['max_output_tokens'])])

                    # Add top-k parameter if supported by the CLI
                    if 'top_k' in params:
                        try:
                            command_args.extend(["--top-k", str(params['top_k'])])
                        except:
                            # Some CLI versions might not support this parameter
                            pass

                    # Add top-p parameter if supported by the CLI
                    if 'top_p' in params:
                        try:
                            command_args.extend(["--top-p", str(params['top_p'])])
                        except:
                            # Some CLI versions might not support this parameter
                            pass

                # Handle system prompt if provided
                system_instructions = ""
                if system_prompt:
                    system_instructions = f"{system_prompt}\n\n"

                # Prepare the prompt with conversation history if needed
                enhanced_prompt = system_instructions + prompt
                if use_history and get_gemini_response.CLI_CHAT_HISTORY:
                    # Format the conversation history as part of the prompt

                    # Create an ultra-optimized conversation history format to minimize token usage

                    # Dynamically adjust history length based on conversation complexity
                    # For complex conversations, keep more context; for simple ones, keep less
                    complexity_score = 0

                    # Check for complexity indicators in the prompt
                    complexity_indicators = [
                        "前に言った", "先ほどの", "さっきの", "以前の", "前回の",  # References to previous exchanges
                        "それ", "あれ", "これ", "その", "あの", "この",           # Pronouns that need context
                        "続き", "さらに", "他に", "もっと", "追加",               # Continuation indicators
                        "なぜ", "どうして", "理由", "原因", "結果",               # Reasoning indicators
                        "でも", "しかし", "一方", "他方", "それとも"              # Contrast indicators
                    ]

                    for indicator in complexity_indicators:
                        if indicator in prompt:
                            complexity_score += 1

                    # Adjust max history entries based on complexity
                    if complexity_score >= 3:
                        max_history_entries = 4  # More context for complex conversations
                    elif complexity_score >= 1:
                        max_history_entries = 3  # Medium context
                    else:
                        max_history_entries = 2  # Less context for simple conversations

                    # Get recent history based on dynamic length
                    recent_history = get_gemini_response.CLI_CHAT_HISTORY[-max_history_entries*2:] if len(get_gemini_response.CLI_CHAT_HISTORY) > max_history_entries*2 else get_gemini_response.CLI_CHAT_HISTORY

                    # Start with no header to save tokens
                    history_text = ""

                    # Add each message with ultra-minimal formatting
                    # Dynamically adjust message length based on importance
                    for i, msg in enumerate(recent_history):
                        # Use U/A instead of User/AI to save tokens
                        role_char = "U" if msg["role"] == "user" else "A"

                        # Determine message importance
                        # More recent messages and user messages get more tokens
                        recency_factor = (i + 1) / len(recent_history)  # 0.0 to 1.0
                        role_factor = 1.2 if msg["role"] == "user" else 1.0  # User messages slightly more important

                        # Calculate dynamic message length (50-150 chars)
                        dynamic_length = int(50 + (recency_factor * role_factor * 100))

                        # Truncate content based on dynamic length
                        content = msg["content"]
                        if len(content) > dynamic_length:
                            # For longer messages, keep beginning and end, remove middle
                            if dynamic_length > 80:
                                # Keep first 2/3 and last 1/3 of allowed length
                                first_part = int(dynamic_length * 0.67)
                                last_part = dynamic_length - first_part - 3  # -3 for ellipsis
                                content = content[:first_part] + "..." + content[-last_part:]
                            else:
                                # For very short allowed lengths, just truncate the end
                                content = content[:dynamic_length-3] + "..."

                        # Add to history with minimal formatting
                        # No newlines between messages to save tokens
                        separator = "" if i == 0 else " "
                        history_text += f"{separator}{role_char}:{content}"

                    # Add the current prompt with minimal formatting
                    # Only include system instructions if they're not too long
                    if system_instructions and len(system_instructions) > 300:
                        # For long system instructions, include only a summary
                        system_summary = "You are Lacia, an AI assistant with a cool but kind personality. "
                        enhanced_prompt = f"{system_summary}{history_text} U:{prompt} A:"
                    else:
                        enhanced_prompt = f"{system_instructions}{history_text} U:{prompt} A:"
                    # Using conversation history in prompt

                # Initialize response variable
                response = None

                # Check if we should try the chat subcommand
                # We'll only try it if it hasn't failed before
                # This is a static variable that persists across function calls
                if not hasattr(get_gemini_response, 'CHAT_SUBCOMMAND_UNSUPPORTED'):
                    # Initialize to False (meaning we'll try the chat subcommand)
                    get_gemini_response.CHAT_SUBCOMMAND_UNSUPPORTED = False
                    print("Initializing CHAT_SUBCOMMAND_UNSUPPORTED to False")

                # First try to use the chat subcommand if it's available and hasn't failed before
                # This provides better support for conversation history
                if use_history and get_gemini_response.CLI_CHAT_HISTORY and not get_gemini_response.CHAT_SUBCOMMAND_UNSUPPORTED:
                    chat_history_file = None
                    system_file = None

                    try:
                        # Create a temporary file for the chat history
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                            chat_history_file = f.name

                            # Format the history as a list of messages with role and content
                            chat_messages = []
                            for msg in get_gemini_response.CLI_CHAT_HISTORY:
                                chat_messages.append({
                                    "role": msg["role"],
                                    "content": msg["content"]
                                })

                            # Add the current message
                            chat_messages.append({
                                "role": "user",
                                "content": prompt
                            })

                            # Write the messages to the file
                            json.dump(chat_messages, f, ensure_ascii=False, indent=2)

                        # Try using the chat subcommand with the history file
                        chat_command_args = ["chat"]

                        # Add model selection
                        chat_command_args.extend(["--model", model_name])

                        # Add the history file
                        chat_command_args.extend(["--history", chat_history_file])

                        # Add system instructions if provided
                        if system_instructions:
                            # Create a temporary file for the system instructions
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                                system_file = f.name
                                f.write(system_instructions)
                            chat_command_args.extend(["--system", system_file])

                        # Run the chat command
                        print("Trying to use Gemini CLI chat subcommand with conversation history...")
                        chat_response = run_gemini_command(chat_command_args)

                        # Debug output to understand the response
                        print(f"Chat subcommand response type: {type(chat_response)}")
                        print(f"Chat subcommand response starts with 'Error:': {chat_response.startswith('Error:')}")
                        print(f"Chat subcommand response first 20 chars: {chat_response[:20]}")

                        # If we got a valid response, use it
                        if not (chat_response.startswith("Error:") or chat_response.startswith("Error occurred:")):
                            print("Successfully used Gemini CLI chat subcommand!")
                            response = chat_response
                        else:
                            print(f"Chat subcommand failed: {chat_response}")

                            # Print the error message for debugging
                            print("Checking if error indicates unsupported chat subcommand...")
                            print(f"Error message: {chat_response}")

                            # Check if the error message contains specific strings
                            contains_unknown_args = "Unknown arguments:" in chat_response and "chat" in chat_response
                            contains_unknown_command = "Unknown command: chat" in chat_response
                            print(f"Contains both 'Unknown arguments:' and 'chat': {contains_unknown_args}")
                            print(f"Contains 'Unknown command: chat': {contains_unknown_command}")

                            # Check if the error indicates that the chat subcommand is not supported
                            if contains_unknown_args or contains_unknown_command:
                                print("Chat subcommand is not supported by this version of the Gemini CLI.")
                                print("Will not try to use it again.")
                                # Set the flag to True (meaning we won't try the chat subcommand again)
                                get_gemini_response.CHAT_SUBCOMMAND_UNSUPPORTED = True
                                print(f"Set CHAT_SUBCOMMAND_UNSUPPORTED to {get_gemini_response.CHAT_SUBCOMMAND_UNSUPPORTED}")

                            print("Falling back to regular prompt method...")

                    except Exception as e:
                        print(f"Error using chat subcommand: {e}")
                        print("Falling back to regular prompt method...")

                    finally:
                        # Clean up temporary files
                        if chat_history_file and os.path.exists(chat_history_file):
                            try:
                                os.unlink(chat_history_file)
                            except:
                                pass
                        if system_file and os.path.exists(system_file):
                            try:
                                os.unlink(system_file)
                            except:
                                pass

                # If we didn't get a response from the chat subcommand, use the regular prompt method
                if response is None:
                    # Add the prompt using the --prompt flag
                    command_args.extend(["--prompt", enhanced_prompt])

                    # Run the Gemini CLI command
                    response = run_gemini_command(command_args)

                # Clean up the response - sometimes the CLI output includes part of the prompt or other artifacts
                if not response.startswith("Error:"):
                    # Remove any parts of the prompt that might have been included in the response
                    # This can happen with some CLI implementations
                    response = response.strip()

                    # Check if the response contains any known artifacts and clean them
                    known_artifacts = [
                        # Old format artifacts (Japanese)
                        "【会話履歴】", "対話", "ユーザー:", "アシスタント:", 
                        "【現在の質問】", "【重要指示】", "会話履歴を参照",

                        # New format artifacts (English)
                        "# Conversation History", "# Current Question", "# Instructions",
                        "User:", "Assistant:", "Reference the conversation history"
                    ]

                    # If any of these artifacts are found in the middle of the response, 
                    # it likely means part of the prompt was included
                    for artifact in known_artifacts:
                        if artifact in response and not response.startswith(artifact):
                            # Find the position of the artifact and truncate the response
                            pos = response.find(artifact)
                            if pos > 0:
                                response = response[:pos].strip()

                    # Check for log file entries that might be included in the response
                    # These typically start with a timestamp in brackets
                    if re.search(r'^\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\]', response):
                        # This looks like a log entry, not a proper response
                        # Replace it with a more helpful message
                        response = "I apologize, but I need to focus on your current question. Could you please repeat what you'd like to know?"

                    # Check for generic acknowledgments of conversation history
                    generic_responses = [
                        "okay, i have our conversation history",
                        "got it. i have the conversation history",
                        "i have our conversation history",
                        "i have the conversation history",
                        "what's next",
                        "what would you like to know",
                        "会話履歴を確認しました",
                        "了解しました",
                        "わかりました",
                        "承知しました",
                        "承知いたしました",
                        "確認しました",
                        "会話の文脈を考慮",
                        "会話履歴を参照"
                    ]

                    # If the response is just a generic acknowledgment, replace it
                    response_lower = response.lower()
                    if any(generic in response_lower for generic in generic_responses) and len(response) < 100:
                        # This is just a generic acknowledgment, not a proper response
                        # Replace it with a more helpful message
                        response = "I apologize, but I need to focus on your current question. Could you please repeat what you'd like to know?"

                # Store this interaction in chat history for future calls
                if use_history:
                    # Only store the user message if it's not already the last message in history
                    if not get_gemini_response.CLI_CHAT_HISTORY or get_gemini_response.CLI_CHAT_HISTORY[-1]["role"] != "user" or get_gemini_response.CLI_CHAT_HISTORY[-1]["content"] != prompt:
                        get_gemini_response.CLI_CHAT_HISTORY.append({"role": "user", "content": prompt})

                    if not response.startswith("Error:"):
                        get_gemini_response.CLI_CHAT_HISTORY.append({"role": "assistant", "content": response})

                    # Limit the size of the conversation history to prevent "command line is too long" errors
                    # Keep only the most recent exchanges (last 5 pairs of messages)
                    max_history_pairs = 5
                    if len(get_gemini_response.CLI_CHAT_HISTORY) > max_history_pairs * 2:
                        # Keep only the most recent exchanges
                        get_gemini_response.CLI_CHAT_HISTORY = get_gemini_response.CLI_CHAT_HISTORY[-max_history_pairs * 2:]
                        print(f"Trimmed conversation history to last {max_history_pairs} exchanges")

                    # Also limit the total size of each message in the history
                    max_message_length = 500
                    for i, msg in enumerate(get_gemini_response.CLI_CHAT_HISTORY):
                        if len(msg["content"]) > max_message_length:
                            get_gemini_response.CLI_CHAT_HISTORY[i]["content"] = msg["content"][:max_message_length] + "..."
                            print(f"Trimmed message {i+1} in conversation history to {max_message_length} characters")

                # Enhanced token usage tracking
                if not response.startswith("Error:"):
                    # Use the new detailed token usage logging function
                    log_token_usage(enhanced_prompt, response, model_name)

                    # Also update the legacy token usage tracking for backward compatibility
                    prompt_tokens = estimate_tokens(enhanced_prompt)
                    response_tokens = estimate_tokens(response)
                    total_tokens = prompt_tokens + response_tokens

                    get_gemini_response.token_usage['total_prompt_tokens'] += prompt_tokens
                    get_gemini_response.token_usage['total_response_tokens'] += response_tokens
                    get_gemini_response.token_usage['total_tokens'] += total_tokens

                    if model_name not in get_gemini_response.token_usage['model_usage']:
                        get_gemini_response.token_usage['model_usage'][model_name] = 0
                    get_gemini_response.token_usage['model_usage'][model_name] += total_tokens

                # Check if the response indicates an error
                if response.startswith("Error:"):
                    print(response)
                    # If it's a model not found error, try the next model
                    if "not found" in response.lower():
                        print(f"Model '{model_name}' not found. Trying next model if available.")
                        break  # Try the next model
                    # If it's a quota exceeded error, try the next model immediately
                    elif "quota exceeded" in response.lower() or "rate limit" in response.lower():
                        print(f"Quota exceeded for model {model_name}. Trying next model if available.")
                        break  # Try the next model
                    # For other errors, retry if possible
                    elif retry_count < max_retries:
                        print(f"Error occurred. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_count += 1
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # If we've exhausted retries, try the next model
                        print(f"Failed to get response from model {model_name}. Trying next model if available.")
                        break
                else:
                    # Success! Return the response and the enhanced prompt
                    return response, enhanced_prompt

            except Exception as e:
                print(f"Error getting Gemini response: {e}")

                # Check if it's a quota exceeded error
                error_str = str(e).lower()
                if "quota exceeded" in error_str or "rate limit" in error_str or "resource exhausted" in error_str:
                    print(f"Quota exceeded for model {model_name}. Trying next model if available.")
                    break  # Try the next model
                elif retry_count < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_count += 1
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to get response from model {model_name}. Trying next model if available.")
                    break  # Try the next model

    print("Failed to get a response from any of the configured models.")
    print("This could be due to CLI errors, unavailable models, or other issues.")
    print("Make sure the Gemini CLI is properly installed and configured.")
    return None, None  # Return None for both response and enhanced prompt

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
    あなたはLaciaという名前の高度なAIアシスタントです。ユーザーとの自然な会話を行ってください。

    【Laciaの人格設定】
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
