import os
import subprocess
import sys

# Disable Streamlit telemetry
os.environ["STREAMLIT_SERVER_HEADLESS"] = "false"

# Run streamlit directly, avoiding interactive prompts
result = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", r"c:\Users\User\Desktop\DVA dashboard\app.py", "--client.showErrorDetails=false"],
    stdin=subprocess.DEVNULL,
    stdout=None,
    stderr=None
)

# Wait indefinitely
result.wait()
