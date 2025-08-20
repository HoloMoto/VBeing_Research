"""
Test script for the conversation logging functionality in play_voice_with_gemini.py.
This script tests the log_conversation function directly.
"""

import os
from play_voice_with_gemini import log_conversation

def test_log_conversation():
    """Test the log_conversation function."""
    print("Testing conversation logging functionality...")
    
    # Delete the Log.txt file if it exists
    if os.path.exists("Log.txt"):
        os.remove("Log.txt")
        print("Deleted existing Log.txt file.")
    
    # Test logging a conversation
    user_input = "This is a test prompt"
    gemini_response = "This is a test response from Gemini"
    mode = "Test Mode"
    
    log_conversation(user_input, gemini_response, mode)
    
    # Check if the Log.txt file was created
    if os.path.exists("Log.txt"):
        print("Log.txt file was created successfully.")
        
        # Read the contents of the Log.txt file
        with open("Log.txt", "r", encoding="utf-8") as log_file:
            log_content = log_file.read()
            
        # Check if the log contains the expected content
        if user_input in log_content and gemini_response in log_content and mode in log_content:
            print("Log.txt contains the expected content.")
            print("Conversation logging test passed!")
        else:
            print("Log.txt does not contain the expected content.")
            print("Conversation logging test failed!")
    else:
        print("Log.txt file was not created.")
        print("Conversation logging test failed!")

if __name__ == "__main__":
    test_log_conversation()