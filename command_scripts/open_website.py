"""
Open Website Command

This script opens a specified website in the default browser.

Usage:
    python open_website.py <url>
    
Example:
    python open_website.py https://www.google.com
"""

import sys
import webbrowser
import re

def is_valid_url(url):
    """Check if the URL is valid."""
    # Simple URL validation regex
    pattern = re.compile(
        r'^(https?://)?' # http:// or https:// (optional)
        r'([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+' # domain
        r'[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?' # domain
        r'(/[a-zA-Z0-9_-]+)*' # path (optional)
        r'(\?[a-zA-Z0-9_=&]+)?' # query parameters (optional)
    )
    return bool(pattern.match(url))

def open_website(url):
    """Open the specified URL in the default browser."""
    # Add http:// prefix if not present
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Open the URL in the default browser
    webbrowser.open(url)
    print(f"Opening website: {url}")
    return True

if __name__ == "__main__":
    # Check if a URL was provided
    if len(sys.argv) < 2:
        print("Error: No URL provided")
        print("Usage: python open_website.py <url>")
        sys.exit(1)
    
    # Get the URL from command line arguments
    url = sys.argv[1]
    
    # Validate the URL
    if not is_valid_url(url):
        print(f"Error: Invalid URL: {url}")
        sys.exit(1)
    
    # Open the website
    success = open_website(url)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)