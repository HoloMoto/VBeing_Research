import os
from play_voice_with_gemini import get_gemini_response, initialize_models

def test_conversation_history():
    """Test the conversation history functionality in get_gemini_response."""
    print("Testing conversation history functionality...")

    # Reset conversation history to start with a clean slate
    from play_voice_with_gemini import reset_conversation_history
    reset_conversation_history()

    # First message
    print("\n--- First message ---")
    response1, prompt1 = get_gemini_response("Hello, how are you?", chat_history=True)
    print(f"Response: {response1}")

    # Second message that refers to the first with explicit instructions
    print("\n--- Second message (should reference first) ---")
    response2, prompt2 = get_gemini_response(
        "Based on our conversation history, what was the exact question I asked you in my previous message? " +
        "Please quote my exact words from the previous message.", 
        chat_history=True
    )
    print(f"Response: {response2}")

    # Third message to further test context with very explicit instructions
    print("\n--- Third message (should maintain context) ---")
    response3, prompt3 = get_gemini_response(
        "Please provide a detailed summary of our entire conversation so far, including: " +
        "1. My first message to you (quote it exactly) " +
        "2. Your response to my first message (summarize it) " +
        "3. My second message to you (quote it exactly) " +
        "4. Your response to my second message (summarize it) " +
        "Be specific and refer to the actual content of our messages.",
        chat_history=True
    )
    print(f"Response: {response3}")

    # Print the conversation history to verify it's being maintained correctly
    print("\n--- Conversation History ---")
    if hasattr(get_gemini_response, 'CLI_CHAT_HISTORY'):
        for i, msg in enumerate(get_gemini_response.CLI_CHAT_HISTORY):
            print(f"{i+1}. {msg['role']}: {msg['content'][:50]}..." if len(msg['content']) > 50 else f"{i+1}. {msg['role']}: {msg['content']}")
    else:
        print("No conversation history found.")

    print("\nTest completed.")

if __name__ == "__main__":
    test_conversation_history()
