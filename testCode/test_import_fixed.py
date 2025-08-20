"""
Test script to verify that the play_voice_with_gemini.py module can be imported without syntax errors.
"""

try:
    import play_voice_with_gemini
    print("Import successful! No syntax errors found.")
except SyntaxError as e:
    print(f"Syntax error: {e}")
except Exception as e:
    print(f"Other error: {e}")