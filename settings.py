"""
Settings Module for VBeing Research

This module handles saving and loading user preferences such as:
- Selected model
- Selected voice
- Selected mode
- Other user preferences

Settings are stored in a JSON file for persistence between sessions.
"""

import os
import json
import time
from pathlib import Path

# Default settings file location
SETTINGS_FILE = "settings.json"

# Default settings values
DEFAULT_SETTINGS = {
    "model": "gemini-2.5-flash",  # Default model
    "mode": 6,  # Default mode (Text-only mode)
    "voice": {
        "use_clone_voice": False,
        "clone_voice_file": None
    },
    "last_updated": None  # Timestamp when settings were last updated
}

def load_settings():
    """
    Load settings from the settings file.
    If the file doesn't exist, return default settings.
    
    Returns:
        dict: The loaded settings or default settings if file doesn't exist
    """
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                print(f"Settings loaded from {SETTINGS_FILE}")
                return settings
        else:
            print(f"Settings file {SETTINGS_FILE} not found. Using default settings.")
            return DEFAULT_SETTINGS.copy()
    except Exception as e:
        print(f"Error loading settings: {e}")
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """
    Save settings to the settings file.
    
    Args:
        settings (dict): The settings to save
        
    Returns:
        bool: True if settings were saved successfully, False otherwise
    """
    try:
        # Update the last_updated timestamp
        settings["last_updated"] = time.time()
        
        # Save settings to file
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
            
        print(f"Settings saved to {SETTINGS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

def update_settings(key, value):
    """
    Update a specific setting.
    
    Args:
        key (str): The setting key to update
        value: The new value for the setting
        
    Returns:
        dict: The updated settings
    """
    settings = load_settings()
    
    # Handle nested settings
    if '.' in key:
        parts = key.split('.')
        current = settings
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    else:
        settings[key] = value
    
    save_settings(settings)
    return settings

def get_setting(key, default=None):
    """
    Get a specific setting value.
    
    Args:
        key (str): The setting key to get
        default: The default value to return if the key doesn't exist
        
    Returns:
        The setting value or default if not found
    """
    settings = load_settings()
    
    # Handle nested settings
    if '.' in key:
        parts = key.split('.')
        current = settings
        for part in parts:
            if part not in current:
                return default
            current = current[part]
        return current
    else:
        return settings.get(key, default)