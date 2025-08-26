# VBeing_Research System Architecture Class Diagram

## Overview

This document provides class diagrams for the VBeing_Research system architecture. Since the system is primarily implemented using a functional programming approach rather than a class-based structure, these diagrams represent the components and their relationships rather than traditional class hierarchies.

## System Components Diagram

```mermaid
classDiagram
    class WebInterface {
        +Flask app
        +index()
        +initialize()
        +set_mode()
        +set_model()
        +send_message()
        +reset_conversation()
        +get_quota_usage()
        +serve_audio()
        +manage_system_prompts()
    }
    
    class CoreProcessingEngine {
        +run_gemini_command()
        +load_system_prompt()
        +get_gemini_response()
        +process_gemini_response()
        +query_with_retries()
        +select_optimal_model()
        +log_conversation()
        +log_token_usage()
    }
    
    class VoiceDataManagement {
        +load_voice_data()
        +create_voice_data_json()
        +play_audio()
        +update_voice_data_csv()
        +get_next_audio_number()
        +get_available_clone_voices()
    }
    
    class AudioSelectionEngine {
        +get_audio_selection_from_gemini()
        +find_best_match_text()
        +rag_audio_selection()
        +get_best_matching_audio()
        +detect_emotions()
        +categorize_voice_data()
        +calculate_context_similarity()
    }
    
    class SpeechRecognition {
        +listen_for_speech()
    }
    
    class ExternalAPIs {
        +Gemini API
        +Zonos API
    }
    
    WebInterface --> CoreProcessingEngine : uses
    WebInterface --> VoiceDataManagement : uses
    WebInterface --> SpeechRecognition : uses
    CoreProcessingEngine --> ExternalAPIs : interfaces with
    CoreProcessingEngine --> AudioSelectionEngine : uses
    CoreProcessingEngine --> VoiceDataManagement : uses
    AudioSelectionEngine --> VoiceDataManagement : uses
```

## Zonos Integration for Japanese Mode

```mermaid
classDiagram
    class WebInterface {
        +check_zonos_connection()
        +direct_tts()
        +generate_and_play_voice()
    }
    
    class ZonosVoiceGeneration {
        +preprocess_text_for_zonos()
        +generate_zonos_voice_data()
        +save_zonos_voice_data()
        +generate_zonos_voice()
        +test_zonos_connection()
        +validate_zonos_text()
        +detect_language_from_system_prompt()
    }
    
    class ZonosAPI {
        +API Key
        +Models
        +Voice Generation
    }
    
    class SystemPromptManagement {
        +load_mode_specific_system_prompt()
        +get_current_system_prompt()
        +save_current_system_prompt()
        +get_system_prompt_templates()
    }
    
    WebInterface --> ZonosVoiceGeneration : uses
    ZonosVoiceGeneration --> ZonosAPI : interfaces with
    WebInterface --> SystemPromptManagement : uses for mode 9
    ZonosVoiceGeneration --> SystemPromptManagement : uses for language detection
```

## Data Flow for Zonos Voice Generation

```mermaid
sequenceDiagram
    participant User
    participant WebInterface
    participant CoreProcessingEngine
    participant ZonosVoiceGeneration
    participant ZonosAPI
    
    User->>WebInterface: Input text (Japanese)
    WebInterface->>CoreProcessingEngine: Process input
    CoreProcessingEngine->>WebInterface: Generate response
    WebInterface->>ZonosVoiceGeneration: Generate voice for response
    ZonosVoiceGeneration->>ZonosVoiceGeneration: Preprocess text
    ZonosVoiceGeneration->>ZonosVoiceGeneration: Validate text
    ZonosVoiceGeneration->>ZonosVoiceGeneration: Detect language
    ZonosVoiceGeneration->>ZonosAPI: Send request with text
    ZonosAPI->>ZonosVoiceGeneration: Return audio data
    ZonosVoiceGeneration->>ZonosVoiceGeneration: Save audio file
    ZonosVoiceGeneration->>WebInterface: Return audio filename
    WebInterface->>User: Play audio response
```

## Component Relationships

The system uses a functional programming approach with these key relationships:

1. **Web Interface** (`web_interface.py`)
   - Imports and uses functions from the Core Processing Engine
   - Handles HTTP routes and user interactions
   - Manages mode selection, including Zonos mode (Mode 9)

2. **Core Processing Engine** (`play_voice_with_gemini.py`)
   - Contains the main logic for processing user inputs
   - Interfaces with external APIs (Gemini and Zonos)
   - Manages audio selection and generation

3. **Zonos Integration**
   - Functions in both files work together to:
     - Check connection status
     - Preprocess and validate text
     - Generate voice using the Zonos API
     - Save and serve audio files

4. **System Prompt Management**
   - Handles loading and saving system prompts
   - Uses mode-specific templates, including the Japanese Zonos template