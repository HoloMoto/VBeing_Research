"""
Test script for the text-only mode (mode 6) of the play_voice_with_gemini.py script.
This script simulates user input to test the text-only mode without requiring manual interaction.
"""

import subprocess
import time
import sys

def test_text_only_mode():
    """Test the text-only mode of the play_voice_with_gemini.py script."""
    try:
        # Start the script as a subprocess
        process = subprocess.Popen(
            ["python", "play_voice_with_gemini.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Wait for the script to initialize
        time.sleep(5)

        # Select mode 6 (text-only mode)
        process.stdin.write("6\n")
        process.stdin.flush()

        # Wait for the mode to be set
        time.sleep(3)

        # Send a test prompt
        process.stdin.write("hello\n")
        process.stdin.flush()

        # Wait for the response
        time.sleep(10)

        # Exit the script
        process.stdin.write("quit\n")
        process.stdin.flush()

        # Wait for the script to exit
        process.wait(timeout=30)

        # Get the output
        stdout, stderr = process.communicate()

        # Print the output
        print("Exit code:", process.returncode)
        print("Stdout:", stdout)
        print("Stderr:", stderr)

        # Check if the error message appears in the output
        if "Unknown arguments: chat" in stdout or "Unknown arguments: chat" in stderr:
            print("Test failed! The error 'Unknown arguments: chat' still appears.")
        else:
            print("Test passed! The error 'Unknown arguments: chat' does not appear.")

    except Exception as e:
        print(f"Error during test: {e}")
        if process:
            process.kill()

if __name__ == "__main__":
    test_text_only_mode()
