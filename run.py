import subprocess
import sys
import os
import threading
import webbrowser
import time

def run_main():
    subprocess.run([sys.executable, "main.py"] + sys.argv[1:])

def run_dashboard():
    time.sleep(2)
    webbrowser.open("http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dashboard":
        webbrowser.open("http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    else:
        print("🚀 Running main.py...")
        main_thread = threading.Thread(target=run_main)
        dashboard_thread = threading.Thread(target=run_dashboard)
        main_thread.start()
        dashboard_thread.start()
        main_thread.join()