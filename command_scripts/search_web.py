"""
Search Web Command

This script searches the web with a specified query using the default browser.

Usage:
    python search_web.py <search_query>
    
Example:
    python search_web.py "weather in Tokyo"
"""

import sys
import webbrowser
import urllib.parse

def search_web(query, search_engine="google"):
    """
    Search the web with the specified query using the specified search engine.
    
    Args:
        query (str): The search query
        search_engine (str): The search engine to use (default: "google")
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Encode the query for URL
    encoded_query = urllib.parse.quote_plus(query)
    
    # Define search URLs for different engines
    search_urls = {
        "google": f"https://www.google.com/search?q={encoded_query}",
        "bing": f"https://www.bing.com/search?q={encoded_query}",
        "yahoo": f"https://search.yahoo.com/search?p={encoded_query}",
        "duckduckgo": f"https://duckduckgo.com/?q={encoded_query}"
    }
    
    # Get the URL for the specified search engine (default to Google if not found)
    url = search_urls.get(search_engine.lower(), search_urls["google"])
    
    # Open the URL in the default browser
    webbrowser.open(url)
    print(f"Searching for '{query}' using {search_engine.capitalize()}")
    return True

if __name__ == "__main__":
    # Check if a search query was provided
    if len(sys.argv) < 2:
        print("Error: No search query provided")
        print("Usage: python search_web.py <search_query> [search_engine]")
        sys.exit(1)
    
    # Get the search query from command line arguments
    query = sys.argv[1]
    
    # Get the search engine if provided (default to Google)
    search_engine = "google"
    if len(sys.argv) > 2:
        search_engine = sys.argv[2]
    
    # Search the web
    success = search_web(query, search_engine)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)