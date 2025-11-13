import subprocess
import os
import sys
import platform
import wx
pythonPath = sys.executable

# --- Path: Absolute path to the main script of your app ---
current_dir = os.path.dirname(__file__)
main_script = os.path.join(current_dir, "app.py")

# --- Check if the file exists ---
if not os.path.exists(main_script):
    import wx
    wx.MessageBox(
        f"Main script not found:\n{main_script}",
        "Error",
        wx.ICON_ERROR
    )
    sys.exit(1)

# --- Run the wxPython app ---
try:
    # If Fiji has the pythonPath variable set, we use it
    subprocess.Popen([pythonPath, main_script])
except Exception as e:
    wx.MessageBox(
        f"Error running the application:\n{str(e)}",
        "Error",
        wx.ICON_ERROR
    )