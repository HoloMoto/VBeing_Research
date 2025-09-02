"""
Command Handler for PC Control

This script handles commands from Gemini to control the PC. It parses Gemini's responses
to extract command strings and executes the appropriate actions.

Usage:
    Import this module and use the execute_command function to run commands from Gemini.

Example:
    from command_handler import execute_command
    
    # Execute a command from Gemini's response
    result = execute_command("run_command('start_edge')")
"""

import os
import re
import subprocess
import ctypes
import time
from pathlib import Path
import pyautogui
from datetime import datetime

# Directory for external command scripts
SCRIPTS_DIR = "command_scripts"

# Ensure the scripts directory exists
os.makedirs(SCRIPTS_DIR, exist_ok=True)

def parse_command(response):
    """
    Parse a response from Gemini to extract command strings.
    
    Args:
        response (str): The response from Gemini
        
    Returns:
        list: A list of (command, args) tuples extracted from the response
    """
    # Regular expression to match run_command('command_name') or run_command('command_name', 'arg1', 'arg2', ...)
    pattern = r"run_command\('([^']+)'(?:\s*,\s*'([^']*)'\s*)*\)"
    
    commands = []
    for match in re.finditer(pattern, response):
        command = match.group(1)
        
        # Extract arguments if present
        args = []
        if match.lastindex and match.lastindex > 1:
            for i in range(2, match.lastindex + 1):
                if match.group(i):
                    args.append(match.group(i))
        
        commands.append((command, args))
    
    return commands

def execute_command(response):
    """
    Execute commands extracted from a Gemini response.
    
    Args:
        response (str): The response from Gemini
        
    Returns:
        dict: A dictionary with the execution results for each command
    """
    commands = parse_command(response)
    results = {}
    
    for command, args in commands:
        try:
            # Check if there's a dedicated script for this command
            script_path = os.path.join(SCRIPTS_DIR, f"{command}.py")
            
            if os.path.exists(script_path):
                # Execute the dedicated script with arguments
                cmd = ["python", script_path] + args
                result = subprocess.run(cmd, capture_output=True, text=True)
                success = result.returncode == 0
                output = result.stdout if success else result.stderr
            else:
                # Handle built-in commands
                success, output = handle_builtin_command(command, args)
            
            results[command] = {
                "success": success,
                "output": output
            }
            
        except Exception as e:
            results[command] = {
                "success": False,
                "output": f"Error executing command: {str(e)}"
            }
    
    return results

def handle_builtin_command(command, args):
    """
    Handle built-in commands that don't require external scripts.
    
    Args:
        command (str): The command to execute
        args (list): Arguments for the command
        
    Returns:
        tuple: (success, output) where success is a boolean and output is a string
    """
    try:
        if command == "start_edge":
            subprocess.Popen(["start", "microsoft-edge:"], shell=True)
            return True, "Edge browser started successfully"
            
        elif command == "start_chrome":
            subprocess.Popen(["start", "chrome"], shell=True)
            return True, "Chrome browser started successfully"
            
        elif command == "start_notepad":
            subprocess.Popen(["notepad"], shell=True)
            return True, "Notepad started successfully"
            
        elif command == "start_calculator":
            subprocess.Popen(["calc"], shell=True)
            return True, "Calculator started successfully"
            
        elif command == "open_file":
            if not args:
                return False, "No file path provided"
            file_path = args[0]
            subprocess.Popen(["start", "", file_path], shell=True)
            return True, f"File opened: {file_path}"
            
        elif command == "create_folder":
            if not args:
                return False, "No folder path provided"
            folder_path = args[0]
            os.makedirs(folder_path, exist_ok=True)
            return True, f"Folder created: {folder_path}"
            
        elif command == "shutdown_pc":
            # Ask for confirmation before shutting down
            confirmation = input("Are you sure you want to shut down the PC? (y/n): ")
            if confirmation.lower() == 'y':
                subprocess.Popen(["shutdown", "/s", "/t", "10"], shell=True)
                return True, "PC will shut down in 10 seconds"
            else:
                return False, "Shutdown cancelled"
            
        elif command == "restart_pc":
            # Ask for confirmation before restarting
            confirmation = input("Are you sure you want to restart the PC? (y/n): ")
            if confirmation.lower() == 'y':
                subprocess.Popen(["shutdown", "/r", "/t", "10"], shell=True)
                return True, "PC will restart in 10 seconds"
            else:
                return False, "Restart cancelled"
            
        elif command == "volume_up":
            # Simulate pressing the volume up key
            pyautogui.press('volumeup')
            return True, "Volume increased"
            
        elif command == "volume_down":
            # Simulate pressing the volume down key
            pyautogui.press('volumedown')
            return True, "Volume decreased"
            
        elif command == "mute":
            # Simulate pressing the mute key
            pyautogui.press('volumemute')
            return True, "Audio muted"
            
        elif command == "take_screenshot":
            # Take a screenshot and save it to the desktop
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(desktop, f"screenshot_{timestamp}.png")
            
            screenshot = pyautogui.screenshot()
            screenshot.save(screenshot_path)
            
            return True, f"Screenshot saved to {screenshot_path}"
            
        else:
            return False, f"Unknown command: {command}"
            
    except Exception as e:
        return False, f"Error executing command {command}: {str(e)}"

if __name__ == "__main__":
    # Test the command handler
    test_response = "Let me help you with that. run_command('start_notepad')"
    results = execute_command(test_response)
    
    for command, result in results.items():
        print(f"Command: {command}")
        print(f"Success: {result['success']}")
        print(f"Output: {result['output']}")
        print()