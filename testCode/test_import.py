try:
    import play_voice_with_gemini
    print("Import successful! The syntax error has been fixed.")
except SyntaxError as e:
    print(f"Syntax error still exists: {e}")
except Exception as e:
    print(f"Other error: {e}")