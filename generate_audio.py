"""
generate_audio.py - Run this ONCE to create new wav files needed for NotGym.
Uses Windows SAPI TTS (same voice as the rest of the app).
Run from the NotGym folder: py -3.11 generate_audio.py
"""
import subprocess
import os
import sys

# Files to generate: (filename, text to speak)
NEW_FILES = [
    # Final 5-sec countdown after Ready/Start Now
    ("prep_5.wav",   "Five"),
    ("prep_4.wav",   "Four"),
    ("prep_3.wav",   "Three"),
    ("prep_2.wav",   "Two"),
    ("prep_1.wav",   "One"),
    ("prep_go.wav",  "Go!"),
    # Prep phase announcements (20s countdown milestones)
    ("get_ready.wav",    "Get into position!"),
    ("say_ready.wav",    "Say Ready when you are set"),
    # Prep phase milestone announcements (replaces exercise milestone wavs during warmup)
    ("prep_ann_15.wav",  "15 seconds"),
    ("prep_ann_10.wav",  "10 seconds"),
    ("prep_ann_5.wav",   "5 seconds. Get ready!"),
    ("starting_now.wav", "Starting now!"),
]

def generate_wav(filename, text):
    """Use PowerShell SAPI to synthesise speech and save as wav."""
    path = os.path.abspath(filename)
    ps_script = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{path}'); "
        f"$s.Speak('{text}'); "
        f"$s.SetOutputToDefaultAudioDevice()"
    )
    result = subprocess.run(
        ["powershell", "-WindowStyle", "Hidden", "-Command", ps_script],
        capture_output=True, text=True
    )
    if result.returncode == 0 and os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  OK  {filename}  ({size} bytes)  \"{text}\"")
        return True
    else:
        print(f"  FAIL  {filename}  -- {result.stderr.strip()}")
        return False

def main():
    print("NotGym Audio Generator")
    print("=" * 40)
    ok = 0
    fail = 0
    for filename, text in NEW_FILES:
        if os.path.exists(filename):
            print(f"  SKIP {filename}  (already exists)")
            ok += 1
        else:
            if generate_wav(filename, text):
                ok += 1
            else:
                fail += 1
    print("=" * 40)
    print(f"Done: {ok} OK, {fail} failed")
    if fail == 0:
        print("\nAll audio files ready. You can now run activeai.py normally.")
    else:
        print("\nSome files failed. Check PowerShell is available.")

if __name__ == "__main__":
    main()
