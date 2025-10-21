import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up paths
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def run_streamlit_app():
    """Run the Streamlit app."""
    # Import here to ensure environment is set up first
    import streamlit.web.bootstrap as bootstrap
    
    # Path to the Streamlit app
    app_path = project_root / "app" / "streamlit_app.py"
    
    # Check if the file exists
    if not app_path.exists():
        print(f"Error: Could not find Streamlit app at {app_path}")
        sys.exit(1)
    
    # Run the app
    print(f"Starting Streamlit app from {app_path}")
    args = ["streamlit", "run", str(app_path), "--browser.serverAddress=localhost", "--server.port=8501"]
    sys.argv = args
    bootstrap.run(args, "", "", "")

if __name__ == "__main__":
    run_streamlit_app() 