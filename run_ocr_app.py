import subprocess


def run_app():
    subprocess.run(["streamlit", "run", "ocr_app.py"])