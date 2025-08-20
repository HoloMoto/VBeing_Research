import subprocess

def test_gemini_cli():
    """Test the Gemini CLI command with the chat subcommand and prompt flag."""
    try:
        # Path to the Gemini CLI executable
        gemini_cli_path = "C:\\Users\\seiri\\AppData\\Roaming\\npm\\gemini.cmd"

        # Test with model and prompt flags (no chat subcommand)
        command = [gemini_cli_path, "--model", "gemini-2.5-pro", "--prompt", "hello"]

        # Run the command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # Don't raise an exception on non-zero exit
            encoding="utf-8"
        )

        # Print the output
        print("Command:", " ".join(command))
        print("Exit code:", result.returncode)
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)

        # Check if the command was successful
        if result.returncode == 0:
            print("Test passed! The command executed successfully.")
        else:
            print("Test failed! The command returned a non-zero exit code.")

    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_gemini_cli()
