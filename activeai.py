"""
plank_server.py  —  Flask backend for AI Plank Trainer
=======================================================
Streams the annotated webcam feed as MJPEG.
Exposes /api/stats for live session data.
Exposes /api/start, /api/reset, /api/end for session control.

Install:
  py -3.11 -m pip install opencv-python mediapipe numpy flask vosk sounddevice

Run:
  py -3.11 plank_server.py

Then open:  http://localhost:5000
"""

import cv2
import numpy as np
import math
import time
import os
import sys
import csv
import json
import threading
import queue
import subprocess
import urllib.request
import zipfile
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory

# ─── VOSK DLL PATH FIX (must be before vosk import anywhere) ──────────────────
# When running as EXE, vosk DLL is in _MEIPASS. When running as .py, use site-packages.
if hasattr(sys, '_MEIPASS'):
    _VOSK_DIR = os.path.join(sys._MEIPASS, 'vosk')
else:
    _VOSK_DIR = r'C:\Users\graja\AppData\Local\Programs\Python\Python311\Lib\site-packages\vosk'
if os.path.isdir(_VOSK_DIR) and _VOSK_DIR not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _VOSK_DIR + os.pathsep + os.environ.get('PATH', '')

# ─── PYINSTALLER RESOURCE PATH ────────────────────────────────────────────────
def resource_path(relative_path):
    """Get absolute path to resource — works for dev and PyInstaller EXE."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ─── VOSK OFFLINE SPEECH RECOGNITION ─────────────────────────────────────────
VOSK_MODEL_DIR  = "vosk-model-small-en-us"
VOSK_MODEL_URL  = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_ZIP  = "vosk-model-small-en-us-0.15.zip"

def ensure_vosk_model():
    """Download and unzip vosk model if not present. Returns model path or None."""
    # When running as EXE, model is bundled in _MEIPASS
    if hasattr(sys, '_MEIPASS'):
        model_path = os.path.join(sys._MEIPASS, VOSK_MODEL_DIR)
        if os.path.isdir(model_path):
            return model_path
    # When running as .py, model sits next to the script
    model_path = os.path.join(os.path.abspath("."), VOSK_MODEL_DIR)
    if os.path.isdir(model_path):
        return model_path
    # Not found — download it
    zip_path = os.path.join(os.path.abspath("."), VOSK_MODEL_ZIP)
    if not os.path.exists(zip_path):
        print("Downloading vosk speech model (~50MB)...")
        try:
            urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path)
            print("Vosk model downloaded.")
        except Exception as e:
            print(f"Failed to download vosk model: {e}")
            return None
    print("Extracting vosk model...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Extract to parent dir — zip contains vosk-model-small-en-us-0.15/
            z.extractall(os.path.dirname(zip_path) or ".")
        # Rename extracted folder to our expected name if needed
        extracted = resource_path("vosk-model-small-en-us-0.15")
        if os.path.isdir(extracted) and not os.path.isdir(model_path):
            os.rename(extracted, model_path)
        os.remove(zip_path)
        print("Vosk model ready.")
        return model_path
    except Exception as e:
        print(f"Failed to extract vosk model: {e}")
        return None

def voice_listener_loop():
    """
    Background thread — continuously listens on default mic using vosk.
    Only triggers action when state.status == 'preparing'.
    """
    try:
        if not hasattr(sys, '_MEIPASS'):
            # Running as .py — ensure site-packages is on path
            _sp = r'C:\Users\graja\AppData\Local\Programs\Python\Python311\Lib\site-packages'
            if os.path.isdir(_sp) and _sp not in sys.path:
                sys.path.insert(0, _sp)
            _vosk_dir = os.path.join(_sp, 'vosk')
            if os.path.isdir(_vosk_dir) and _vosk_dir not in os.environ.get('PATH', ''):
                os.environ['PATH'] = _vosk_dir + os.pathsep + os.environ.get('PATH', '')
        import vosk
        import sounddevice as sd
    except Exception as e:
        print(f"vosk/sounddevice import failed: {type(e).__name__}: {e}")
        return

    model_path = ensure_vosk_model()
    if not model_path:
        print("Vosk model unavailable — voice commands disabled.")
        return

    try:
        vosk.SetLogLevel(-1)   # suppress vosk logging
        model = vosk.Model(model_path)
    except Exception as e:
        print(f"Failed to load vosk model: {e}")
        return

    SAMPLE_RATE = 16000
    BLOCK_SIZE  = 4000   # ~250ms chunks

    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(False)

    print("Voice listener ready. Say 'Ready' during prep countdown to start.")

    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                               dtype='int16', channels=1) as stream:
            while True:
                data, _ = stream.read(BLOCK_SIZE)
                with state.lock:
                    listening = state.status == 'preparing'
                if not listening:
                    rec.Reset()
                    continue

                if rec.AcceptWaveform(bytes(data)):
                    result = json.loads(rec.Result())
                    text = result.get('text', '').lower().strip()
                    if 'ready' in text:
                        print(f"[vosk] 'Ready' detected → starting session")
                        _begin_active_session()
                        rec.Reset()
                else:
                    partial = json.loads(rec.PartialResult())
                    text = partial.get('partial', '').lower().strip()
                    if 'ready' in text:
                        print(f"[vosk] 'Ready' detected (partial) → starting session")
                        _begin_active_session()
                        rec.Reset()
    except Exception as e:
        print(f"Voice listener error: {e}")

# ─── MODEL ────────────────────────────────────────────────────────────────────
MODEL_FILE = "pose_landmarker_lite.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

def ensure_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading pose model (~5 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Model ready.")

# ─── AUDIO ────────────────────────────────────────────────────────────────────
_VOICE_FILES = {
    "ready":          "ready.wav",
    "good":           "good_form.wav",
    "warn_hips_low":  "hips_low.wav",
    "warn_hips_high": "hips_high.wav",
    "warn_head_low":  "head_low.wav",
    "warn_head_high": "head_high.wav",
    "warn_elbows":    "elbows.wav",
    "reset":          "reset.wav",
    "paused":         "paused.wav",
    "done":           "done.wav",
    5:   "5sec.wav",   10:  "10sec.wav",  15:  "15sec.wav",
    20:  "20sec.wav",  25:  "25sec.wav",  30:  "30sec.wav",
    45:  "45sec.wav",  60:  "60sec.wav",  90:  "90sec.wav",
    120: "120sec.wav",
    # Prep countdown (after Ready/Start Now) — 5..1..Go
    "prep_5":   "prep_5.wav",
    "prep_4":   "prep_4.wav",
    "prep_3":   "prep_3.wav",
    "prep_2":   "prep_2.wav",
    "prep_1":   "prep_1.wav",
    "prep_go":  "prep_go.wav",
    # Prep phase announcements
    "get_ready": "get_ready.wav",
    "say_ready": "say_ready.wav",
    # Prep phase milestone announcements (countdown-specific, not exercise milestones)
    "prep_ann_15": "prep_ann_15.wav",
    "prep_ann_10": "prep_ann_10.wav",
    "prep_ann_5":  "prep_ann_5.wav",
    "starting_now": "starting_now.wav",
    # Squat rep milestones
    "sq_5":  "sq_5reps.wav",
    "sq_10": "sq_10reps.wav",
    "sq_15": "sq_15reps.wav",
    "sq_20": "sq_20reps.wav",
    "sq_25": "sq_25reps.wav",
    "sq_30": "sq_30reps.wav",
    "sq_40": "sq_40reps.wav",
    "sq_50": "sq_50reps.wav",
    # Squat form warnings
    "sq_lean":  "sq_lean.wav",
    "sq_depth": "sq_depth.wav",
    # Squat individual rep counts (1-9 excluding 5)
    "sq_rep1": "sq_rep1.wav",
    "sq_rep2": "sq_rep2.wav",
    "sq_rep3": "sq_rep3.wav",
    "sq_rep4": "sq_rep4.wav",
    "sq_rep6": "sq_rep6.wav",
    "sq_rep7": "sq_rep7.wav",
    "sq_rep8": "sq_rep8.wav",
    "sq_rep9": "sq_rep9.wav",
    # Pushup rep milestones (reuse squat files)
    "pu_5":  "sq_5reps.wav",
    "pu_10": "sq_10reps.wav",
    "pu_15": "sq_15reps.wav",
    "pu_20": "sq_20reps.wav",
    "pu_25": "sq_25reps.wav",
    "pu_30": "sq_30reps.wav",
    "pu_40": "sq_40reps.wav",
    "pu_50": "sq_50reps.wav",
    # Pushup form warnings
    "pu_sag":   "pu_sag.wav",
    "pu_pike":  "pu_pike.wav",
    "pu_flare": "pu_flare.wav",
    # Pushup individual reps
    "pu_rep1": "sq_rep1.wav",
    "pu_rep2": "sq_rep2.wav",
    "pu_rep3": "sq_rep3.wav",
    "pu_rep4": "sq_rep4.wav",
    "pu_rep6": "sq_rep6.wav",
    "pu_rep7": "sq_rep7.wav",
    "pu_rep8": "sq_rep8.wav",
    "pu_rep9": "sq_rep9.wav",
}
_FALLBACK_WAV = r"C:\Windows\Media\ding.wav"

def _resolve_wav(name):
    fname = _VOICE_FILES.get(name, "")
    if fname:
        full_path = resource_path(fname)
        if os.path.exists(full_path):
            return full_path
    return _FALLBACK_WAV

class AudioCues:
    def __init__(self):
        self.enabled = True
        self._q      = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        current_proc = None
        while True:
            try:
                name = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if name is None:
                if current_proc and current_proc.poll() is None:
                    current_proc.kill()
                break
            if name == '__flush__':
                # Kill current playback, drain queue
                if current_proc and current_proc.poll() is None:
                    current_proc.kill()
                    current_proc = None
                while not self._q.empty():
                    try: self._q.get_nowait()
                    except queue.Empty: break
                continue
            if self.enabled:
                if current_proc and current_proc.poll() is None:
                    current_proc.kill()
                path = _resolve_wav(name)
                ps = f"(New-Object Media.SoundPlayer '{path}').PlaySync()"
                try:
                    current_proc = subprocess.Popen(
                        ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    current_proc.wait(timeout=8)
                except Exception:
                    pass

    def play(self, name, priority=False):
        if not self.enabled:
            return
        if priority:
            while not self._q.empty():
                try: self._q.get_nowait()
                except queue.Empty: break
        self._q.put(name)

    def flush(self):
        """Stop current audio and clear queue — call at session start."""
        self._q.put('__flush__')

    def stop(self):
        self._q.put(None)

# ─── SESSION LOG ──────────────────────────────────────────────────────────────
LOG_FILE = "plank_log.csv"

def fmt_time(s):
    return f"{int(s)//60:02d}:{int(s)%60:02d}"

def save_session(best_hold_secs, name="", rep_count=0, exercise="plank"):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["name","exercise","date","time","best_hold_sec","best_hold_fmt","reps"])
        now = datetime.now()
        writer.writerow([
            name,
            exercise,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M"),
            round(best_hold_secs, 1),
            fmt_time(best_hold_secs),
            rep_count,
        ])

def load_log():
    if not os.path.exists(LOG_FILE):
        return []
    rows = []
    with open(LOG_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# ─── LANDMARKS ────────────────────────────────────────────────────────────────
NOSE=0; LEFT_SHOULDER=11; RIGHT_SHOULDER=12
LEFT_ELBOW=13; RIGHT_ELBOW=14
LEFT_HIP=23; RIGHT_HIP=24
LEFT_KNEE=25; RIGHT_KNEE=26
LEFT_ANKLE=27; RIGHT_ANKLE=28

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
]

# ─── CONFIG ───────────────────────────────────────────────────────────────────
HIP_SAG_THRESH   = 0.04
HIP_PIKE_THRESH  = 0.04
HEAD_THRESH      = 0.06
GOOD_FORM_BUFFER = 0.5
MILESTONE_EVERY  = 5
ISSUE_COOLDOWN   = 4.0
STAND_TO_END     = 10.0

C_GOOD  = (80,  220, 100)
C_WARN  = (30,  180, 255)
C_BAD   = (60,   60, 220)
C_WHITE = (255, 255, 255)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def lm_px(lms, idx, w, h):
    l = lms[idx]; return int(l.x*w), int(l.y*h), l.visibility

def angle_3pts(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
    cos_a = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return math.degrees(math.acos(np.clip(cos_a,-1,1)))

def pt_line_dev(px,py,ax,ay,bx,by,h):
    dx,dy=bx-ax,by-ay; L=math.hypot(dx,dy)+1e-6
    return ((px-ax)*dy-(py-ay)*dx)/L/h

def draw_skeleton(frame, lms, w, h, colour):
    pts={i:(int(lms[i].x*w),int(lms[i].y*h)) for i in range(min(33,len(lms)))}
    for a,b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame,pts[a],pts[b],colour,2)
    for pt in pts.values():
        cv2.circle(frame,pt,3,(220,220,220),-1)

def is_standing(lms, w, h):
    l = lms[LEFT_SHOULDER].visibility+lms[LEFT_ANKLE].visibility
    r = lms[RIGHT_SHOULDER].visibility+lms[RIGHT_ANKLE].visibility
    if l>=r: sx,sy,_=lm_px(lms,LEFT_SHOULDER,w,h); ax,ay,_=lm_px(lms,LEFT_ANKLE,w,h)
    else:    sx,sy,_=lm_px(lms,RIGHT_SHOULDER,w,h); ax,ay,_=lm_px(lms,RIGHT_ANKLE,w,h)
    return math.degrees(math.atan2(abs(ax-sx),abs(ay-sy)+1e-6)) < 30

def analyse_plank(lms, w, h):
    issues=[]
    ls=lms[LEFT_SHOULDER].visibility+lms[LEFT_HIP].visibility+lms[LEFT_ANKLE].visibility
    rs=lms[RIGHT_SHOULDER].visibility+lms[RIGHT_HIP].visibility+lms[RIGHT_ANKLE].visibility
    side='left' if ls>=rs else 'right'
    if side=='left':
        sx,sy,_=lm_px(lms,LEFT_SHOULDER,w,h); hx,hy,_=lm_px(lms,LEFT_HIP,w,h)
        ax,ay,_=lm_px(lms,LEFT_ANKLE,w,h);   ex,ey,_=lm_px(lms,LEFT_ELBOW,w,h)
    else:
        sx,sy,_=lm_px(lms,RIGHT_SHOULDER,w,h); hx,hy,_=lm_px(lms,RIGHT_HIP,w,h)
        ax,ay,_=lm_px(lms,RIGHT_ANKLE,w,h);    ex,ey,_=lm_px(lms,RIGHT_ELBOW,w,h)
    nx,ny,_=lm_px(lms,NOSE,w,h)
    angle=angle_3pts((sx,sy),(hx,hy),(ax,ay))
    hd=pt_line_dev(hx,hy,sx,sy,ax,ay,h)
    nd=pt_line_dev(nx,ny,sx,sy,ax,ay,h)
    ed=pt_line_dev(ex,ey,sx,sy,ax,ay,h)
    if hd>HIP_SAG_THRESH:    issues.append(("Hips too low","warn_hips_low"))
    elif hd<-HIP_PIKE_THRESH: issues.append(("Hips too high","warn_hips_high"))
    if nd>HEAD_THRESH:        issues.append(("Head drooping","warn_head_low"))
    elif nd<-HEAD_THRESH*1.5: issues.append(("Head too high","warn_head_high"))
    if ed<-0.08:              issues.append(("Elbows forward","warn_elbows"))
    colour=C_GOOD if not issues else (C_WARN if len(issues)==1 else C_BAD)
    return issues, angle, colour, side, (sx,sy),(hx,hy),(ax,ay)


# ─── SQUAT ANALYSER ───────────────────────────────────────────────────────────
# Side view. Tracks hip-knee-ankle angle for depth + back angle for form.
# Rep counted when: angle drops below DOWN_THRESH then rises above UP_THRESH.

SQ_UP_THRESH   = 160   # knee angle = standing (rep complete)
SQ_DOWN_THRESH = 100   # knee angle = bottom of squat
SQ_BACK_THRESH = 50    # back angle from vertical — too much forward lean

def analyse_squat(lms, w, h, squat_state, rep_count):
    """
    Returns: issues, knee_angle, back_angle, colour, new_squat_state,
             new_rep_count, side, knee_pt, hip_pt, ankle_pt
    """
    issues = []
    ls = lms[LEFT_HIP].visibility + lms[25].visibility + lms[LEFT_ANKLE].visibility
    rs = lms[RIGHT_HIP].visibility + lms[26].visibility + lms[RIGHT_ANKLE].visibility
    side = 'left' if ls >= rs else 'right'

    if side == 'left':
        sx, sy, _ = lm_px(lms, LEFT_SHOULDER,  w, h)
        hx, hy, _ = lm_px(lms, LEFT_HIP,       w, h)
        kx, ky, _ = lm_px(lms, LEFT_KNEE,      w, h)
        ax, ay, _ = lm_px(lms, LEFT_ANKLE,     w, h)
    else:
        sx, sy, _ = lm_px(lms, RIGHT_SHOULDER, w, h)
        hx, hy, _ = lm_px(lms, RIGHT_HIP,      w, h)
        kx, ky, _ = lm_px(lms, RIGHT_KNEE,     w, h)
        ax, ay, _ = lm_px(lms, RIGHT_ANKLE,    w, h)

    # Knee angle: hip-knee-ankle
    knee_angle = angle_3pts((hx,hy), (kx,ky), (ax,ay))

    # Back angle: angle of shoulder-hip line from vertical
    back_angle = math.degrees(math.atan2(abs(sx-hx), abs(sy-hy)+1e-6))

    # ── Rep counting state machine ────────────────────────────────────────────
    new_state = squat_state
    new_reps  = rep_count

    if squat_state == 'up' and knee_angle < SQ_DOWN_THRESH:
        new_state = 'down'
    elif squat_state == 'down' and knee_angle > SQ_UP_THRESH:
        new_state = 'up'
        new_reps  = rep_count + 1

    # ── Form checks ───────────────────────────────────────────────────────────
    if new_state == 'down':
        if back_angle > SQ_BACK_THRESH:
            issues.append(("Leaning too far forward", "warn_hips_low"))
        if knee_angle > SQ_DOWN_THRESH + 15:
            issues.append(("Go deeper", "warn_hips_high"))

    colour = C_GOOD if not issues else (C_WARN if len(issues)==1 else C_BAD)

    return (issues, knee_angle, back_angle, colour,
            new_state, new_reps, side,
            (kx,ky), (hx,hy), (ax,ay))


# ─── PUSHUP ANALYSER ──────────────────────────────────────────────────────────
# Front view. Tracks average elbow angle for rep counting.
# Also checks body sag (hip below shoulder-ankle line) and elbow flare.

PU_UP_THRESH   = 155   # elbow angle = top of push-up (rep complete)
PU_DOWN_THRESH = 90    # elbow angle = bottom of push-up
PU_FLARE_THRESH = 65   # max degrees elbows can flare from body line

def analyse_pushup(lms, w, h, pu_state, rep_count):
    """
    Returns: issues, elbow_angle, colour, new_pu_state, new_rep_count,
             l_elbow_pt, r_elbow_pt, body_pts
    """
    issues = []

    # Use both sides — front view gives us both
    lsx, lsy, _ = lm_px(lms, LEFT_SHOULDER,  w, h)
    rsx, rsy, _ = lm_px(lms, RIGHT_SHOULDER, w, h)
    lex, ley, _ = lm_px(lms, LEFT_ELBOW,     w, h)
    rex, rey, _ = lm_px(lms, RIGHT_ELBOW,    w, h)
    lwx, lwy, _ = lm_px(lms, 15, w, h)  # LEFT_WRIST
    rwx, rwy, _ = lm_px(lms, 16, w, h)  # RIGHT_WRIST
    lhx, lhy, _ = lm_px(lms, LEFT_HIP,      w, h)
    rhx, rhy, _ = lm_px(lms, RIGHT_HIP,     w, h)
    lax, lay, _ = lm_px(lms, LEFT_ANKLE,    w, h)
    rax, ray, _ = lm_px(lms, RIGHT_ANKLE,   w, h)

    # Average elbow angle from both arms
    l_angle = angle_3pts((lsx,lsy), (lex,ley), (lwx,lwy))
    r_angle = angle_3pts((rsx,rsy), (rex,rey), (rwx,rwy))
    elbow_angle = (l_angle + r_angle) / 2

    # ── Rep counting ──────────────────────────────────────────────────────────
    new_state = pu_state
    new_reps  = rep_count
    if pu_state == 'up' and elbow_angle < PU_DOWN_THRESH:
        new_state = 'down'
    elif pu_state == 'down' and elbow_angle > PU_UP_THRESH:
        new_state = 'up'
        new_reps  = rep_count + 1

    # ── Form checks ───────────────────────────────────────────────────────────
    # 1. Body sag — average hip deviation from shoulder-ankle line (both sides)
    l_sag = pt_line_dev(lhx, lhy, lsx, lsy, lax, lay, h)
    r_sag = pt_line_dev(rhx, rhy, rsx, rsy, rax, ray, h)
    avg_sag = (l_sag + r_sag) / 2
    if avg_sag > HIP_SAG_THRESH:
        issues.append(("Hips sagging — lift core", "pu_sag"))
    elif avg_sag < -HIP_PIKE_THRESH:
        issues.append(("Hips too high", "pu_pike"))

    # 2. Elbow flare — angle of elbow relative to shoulder line
    # Measure how far elbow is from the torso midline
    mid_sx = (lsx + rsx) / 2
    shoulder_width = abs(lsx - rsx) + 1e-6
    l_flare = abs(lex - lsx) / shoulder_width
    r_flare = abs(rex - rsx) / shoulder_width
    if l_flare > 0.6 or r_flare > 0.6:
        issues.append(("Elbows flaring — tuck them in", "pu_flare"))

    colour = C_GOOD if not issues else (C_WARN if len(issues)==1 else C_BAD)
    return (issues, elbow_angle, colour, new_state, new_reps,
            (lex,ley), (rex,rey), (lsx,lsy), (rsx,rsy), (lhx,lhy), (rhx,rhy))


# ─── MOUNTAIN CLIMBER ANALYSER ────────────────────────────────────────────────
# Side or slight front view. Each knee drive toward chest = 1 rep.
# Tracks both legs alternately. Also checks body alignment (no hip sag).

MC_KNEE_THRESH = 0.12   # knee y must be this much higher than hip y (normalised)

def analyse_mountain_climber(lms, w, h, mc_state, rep_count):
    """
    Side view (same camera as plank).
    Picks the more visible side and tracks that knee driving forward.
    Each full knee-forward + knee-back cycle = 1 rep per leg.
    mc_state: {'left': 'down', 'right': 'down'}
    """
    issues = []

    # Pick best visible side (same logic as plank/squat)
    l_score = (lms[LEFT_SHOULDER].visibility +
               lms[LEFT_HIP].visibility +
               lms[LEFT_KNEE].visibility)
    r_score = (lms[RIGHT_SHOULDER].visibility +
               lms[RIGHT_HIP].visibility +
               lms[RIGHT_KNEE].visibility)
    side = 'left' if l_score >= r_score else 'right'

    if side == 'left':
        sx, sy, _ = lm_px(lms, LEFT_SHOULDER, w, h)
        hx, hy, _ = lm_px(lms, LEFT_HIP,      w, h)
        kx, ky, _ = lm_px(lms, LEFT_KNEE,     w, h)
        ax, ay, _ = lm_px(lms, LEFT_ANKLE,    w, h)
        leg_key   = 'left'
    else:
        sx, sy, _ = lm_px(lms, RIGHT_SHOULDER, w, h)
        hx, hy, _ = lm_px(lms, RIGHT_HIP,      w, h)
        kx, ky, _ = lm_px(lms, RIGHT_KNEE,     w, h)
        ax, ay, _ = lm_px(lms, RIGHT_ANKLE,    w, h)
        leg_key   = 'right'

    new_state = mc_state.copy()
    new_reps  = rep_count

    # In side view: knee drives FORWARD past the hip toward the chest
    # Primary signal: knee X passes hip X (clearly forward)
    # Secondary: knee Y rises above hip Y (upward drive)
    knee_forward = (hx - kx) / w if side == 'left' else (kx - hx) / w
    knee_up      = (hy - ky) / h

    # Rep fires when knee is clearly past the hip horizontally
    # OR significantly raised — either signal is enough
    knee_driven = knee_forward > 0.08 or knee_up > 0.10

    if knee_driven and new_state.get(leg_key) == 'down':
        new_state[leg_key] = 'up'
        new_reps += 1
    elif not knee_driven and new_state.get(leg_key) == 'up':
        new_state[leg_key] = 'down'

    # Mountain climber form check:
    # Hips should stay roughly level with shoulders — not bouncing wildly.
    # We measure hip Y relative to shoulder Y, normalised by frame height.
    # In push-up start position, hip is slightly below shoulder (positive diff).
    # Allow a generous band — mountain climber is dynamic, not a static hold.
    MC_HIP_LOW_THRESH  = 0.20   # hip more than 20% of frame below shoulder = sagging
    MC_HIP_HIGH_THRESH = 0.05   # hip more than 5% of frame above shoulder = piking

    hip_rel = (hy - sy) / h   # positive = hip lower than shoulder (normal in pushup)

    if hip_rel > MC_HIP_LOW_THRESH:
        issues.append(("Hips dropping — keep core engaged", "pu_sag"))
    elif hip_rel < -MC_HIP_HIGH_THRESH:
        issues.append(("Hips too high — lower them", "pu_pike"))

    colour = C_GOOD if not issues else C_WARN
    return (issues, colour, new_state, new_reps,
            (sx, sy), (hx, hy), (kx, ky), (kx, ky))


# ─── JUMPING JACK ANALYSER ────────────────────────────────────────────────────
# Front view. Tracks arms AND feet simultaneously.
# Rep = both arms up AND feet wide → both arms down AND feet together = 1 rep.
# Uses a 2-phase state machine: 'closed' and 'open'.

JJ_ARM_THRESH  = 0.04   # wrist must be this much higher than shoulder (normalised y)
JJ_FEET_THRESH = 0.08   # each ankle must be this much wider than hip (normalised x)

def analyse_jumping_jack(lms, w, h, jj_state, rep_count):
    """
    jj_state: 'closed' | 'open'
    Returns: issues, colour, new_jj_state, new_rep_count,
             l_wrist_pt, r_wrist_pt, l_ankle_pt, r_ankle_pt
    """
    issues = []

    lsx, lsy, _ = lm_px(lms, LEFT_SHOULDER,  w, h)
    rsx, rsy, _ = lm_px(lms, RIGHT_SHOULDER, w, h)
    lhx, lhy, _ = lm_px(lms, LEFT_HIP,       w, h)
    rhx, rhy, _ = lm_px(lms, RIGHT_HIP,      w, h)
    lax, lay, _ = lm_px(lms, LEFT_ANKLE,     w, h)
    rax, ray, _ = lm_px(lms, RIGHT_ANKLE,    w, h)
    lwx, lwy, _ = lm_px(lms, 15, w, h)   # LEFT_WRIST
    rwx, rwy, _ = lm_px(lms, 16, w, h)   # RIGHT_WRIST

    # Arms up: both wrists above their respective shoulders
    l_arm_up = (lsy - lwy) / h > JJ_ARM_THRESH
    r_arm_up = (rsy - rwy) / h > JJ_ARM_THRESH
    arms_up  = l_arm_up and r_arm_up

    # Feet wide: ankles outside hips
    l_foot_wide = (lhx - lax) / w > JJ_FEET_THRESH
    r_foot_wide = (rax - rhx) / w > JJ_FEET_THRESH
    feet_wide   = l_foot_wide and r_foot_wide

    # Arms down: wrists below shoulders
    arms_down = (lwy - lsy) / h > 0.02 and (rwy - rsy) / h > 0.02

    # Feet together: ankles close to hips
    feet_together = (lhx - lax) / w < 0.02 and (rax - rhx) / w < 0.02

    new_state = jj_state
    new_reps  = rep_count

    if jj_state == 'closed' and arms_up and feet_wide:
        new_state = 'open'
    elif jj_state == 'open' and arms_down and feet_together:
        new_state = 'closed'
        new_reps  = rep_count + 1

    # Form check — arms should go fully overhead, not just halfway
    if jj_state == 'open' and not arms_up:
        issues.append(("Raise arms fully overhead", "jj_arms"))

    colour = C_GOOD if not issues else C_WARN
    return (issues, colour, new_state, new_reps,
            (lwx,lwy), (rwx,rwy), (lax,lay), (rax,ray))

# ─── SESSION STATE (shared between Flask routes and camera thread) ─────────────
PREP_COUNTDOWN_SECS = 20   # seconds to get ready before exercise tracking starts

class SessionState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.running        = False
        self.player_name    = ""
        self.hold_time      = 0.0
        self.best_hold      = 0.0
        self.status         = "idle"   # idle | preparing | active | standing | ended
        self.feedback       = ""
        self.angle          = 180.0
        self.good_form      = False
        self.standing       = False
        self.countdown      = 0
        self.prep_countdown = 0        # countdown before exercise starts (preparing phase)
        self.latest_frame   = None     # JPEG bytes of latest annotated frame
        self.reset_signal   = False    # True = camera thread should reset local state
        self.exercise       = 'plank'   # plank | squat
        self.rep_count      = 0
        self.squat_state    = 'up'      # up | down (for rep counting)
        self.pu_state       = 'up'       # up | down (for pushup rep counting)
        self.mc_state       = {'left':'down','right':'down'}  # mountain climber leg states
        self.jj_state       = 'closed'   # jumping jack phase

state = SessionState()
audio = AudioCues()

# ─── CAMERA THREAD ────────────────────────────────────────────────────────────
def camera_loop():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    ensure_model()

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        num_poses=1,
    )

    cap = cv2.VideoCapture(0)
    start_ms       = int(time.time()*1000)
    hold_time      = 0.0
    best_hold      = 0.0
    good_form_start= None
    was_good_form  = False
    was_standing   = False
    standing_since = None
    last_milestone = 0
    last_issue_t   = {}
    last_seen      = time.time()
    rep_count      = 0
    squat_state    = 'up'
    pu_state       = 'up'
    mc_state       = {'left':'down','right':'down'}
    last_rep_time  = None  # only set after first rep — prevents premature timeout
    prev_rep_count = 0
    jj_state       = 'closed'

    import traceback
    log = open("camera_log.txt", "w", buffering=1)
    log.write("Camera thread started\n")
    log.flush()

    log.write("Landmarker created OK\n"); log.flush()
    frame_count = 0
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            with state.lock:
                running = state.running
                name    = state.player_name
                if state.reset_signal:
                    # New session started — reset all local tracking vars immediately
                    # (done here regardless of running so stale state never fires audio)
                    hold_time=0.0; best_hold=0.0; good_form_start=None
                    was_good_form=False; was_standing=False
                    standing_since=None; last_milestone=0; last_issue_t={}
                    rep_count=0; squat_state='up'; pu_state='up'
                    mc_state={'left':'down','right':'down'}
                    last_rep_time=None; prev_rep_count=0
                    jj_state='closed'
                    state.reset_signal = False

            if not running:
                time.sleep(0.05)
                # Still capture + encode blank frame so stream doesn't freeze
                ret, frame = cap.read()
                if ret:
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    with state.lock:
                        state.latest_frame = buf.tobytes()
                continue

            ret, frame = cap.read()
            if not ret:
                log.write("cap.read() failed\n"); log.flush()
                time.sleep(0.01)
                continue

            h, w   = frame.shape[:2]
            now    = time.time()
            ts_ms  = int(now*1000) - start_ms

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                result = landmarker.detect_for_video(mp_img, ts_ms)
            except Exception as det_ex:
                log.write(f"detect_for_video FAILED: {det_ex}\n"); log.flush()
                continue

            feedback  = ""
            angle_val = 180.0
            gf        = False
            standing  = False
            countdown = 0

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lms = result.pose_landmarks[0]
                last_seen = now

                # Read current exercise from shared state
                with state.lock:
                    exercise    = state.exercise
                    squat_state = state.squat_state
                    rep_count   = state.rep_count

                standing = is_standing(lms, w, h)

                # ── Route to correct analyser ─────────────────────────────
                if exercise == 'squat':
                    (issues, angle_val, back_angle, status_col,
                     squat_state, rep_count, side,
                     kpt, hpt, apt) = analyse_squat(
                         lms, w, h, squat_state, rep_count)
                    gf = (len(issues) == 0) and not standing

                    draw_skeleton(frame, lms, w, h, status_col)
                    cv2.line(frame, hpt, kpt, status_col, 3)
                    cv2.line(frame, kpt, apt, status_col, 3)
                    cv2.circle(frame, kpt, 10, status_col,    -1)
                    cv2.circle(frame, hpt,  8, (255,200,60),  -1)
                    cv2.circle(frame, apt,  8, (255,200,60),  -1)

                    # Rep counter on frame
                    cv2.putText(frame, f"Reps: {rep_count}",
                                (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255,200,60), 3)
                    cv2.putText(frame,
                                f"Knee: {angle_val:.0f}deg",
                                (12, 68), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (220,220,220), 2)

                    with state.lock:
                        state.squat_state = squat_state
                        state.rep_count   = rep_count

                elif exercise == 'pushup':
                    (issues, elbow_angle, status_col, pu_state, rep_count,
                     lept, rept, lspt, rspt, lhpt, rhpt) = analyse_pushup(
                         lms, w, h, pu_state, rep_count)
                    gf = (len(issues) == 0)
                    angle_val = elbow_angle

                    draw_skeleton(frame, lms, w, h, status_col)
                    # Draw arm lines
                    cv2.line(frame, lspt, lept, status_col, 3)
                    cv2.line(frame, rspt, rept, status_col, 3)
                    cv2.circle(frame, lept, 10, status_col,   -1)
                    cv2.circle(frame, rept, 10, status_col,   -1)
                    cv2.circle(frame, lspt,  8, (255,200,60), -1)
                    cv2.circle(frame, rspt,  8, (255,200,60), -1)

                    cv2.putText(frame, f"Reps: {rep_count}",
                                (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255,200,60), 3)
                    cv2.putText(frame, f"Elbow: {elbow_angle:.0f}deg",
                                (12, 68), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (220,220,220), 2)

                    with state.lock:
                        state.squat_state = pu_state  # reuse field
                        state.pu_state    = pu_state
                        state.rep_count   = rep_count

                elif exercise == 'mountain_climber':
                    with state.lock:
                        mc_state  = state.mc_state.copy()

                    (issues, status_col, mc_state, rep_count,
                     sh_pt, hi_pt, lk_pt, rk_pt) = analyse_mountain_climber(
                         lms, w, h, mc_state, rep_count)
                    gf = (len(issues) == 0)
                    angle_val = 180.0

                    draw_skeleton(frame, lms, w, h, status_col)
                    cv2.line(frame, sh_pt, hi_pt, status_col, 3)
                    cv2.circle(frame, hi_pt, 8, (255,200,60), -1)
                    cv2.circle(frame, lk_pt, 8, status_col, -1)
                    cv2.circle(frame, rk_pt, 8, status_col, -1)

                    cv2.putText(frame, f"Reps: {rep_count}",
                                (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255,200,60), 3)

                    with state.lock:
                        state.mc_state  = mc_state
                        state.rep_count = rep_count

                elif exercise == 'jumping_jack':
                    with state.lock:
                        jj_state = state.jj_state

                    (issues, status_col, jj_state, rep_count,
                     lwpt, rwpt, lapt, rapt) = analyse_jumping_jack(
                         lms, w, h, jj_state, rep_count)
                    gf = (len(issues) == 0)
                    angle_val = 180.0

                    draw_skeleton(frame, lms, w, h, status_col)
                    # Highlight wrists and ankles
                    cv2.circle(frame, lwpt, 8, status_col, -1)
                    cv2.circle(frame, rwpt, 8, status_col, -1)
                    cv2.circle(frame, lapt, 8, (255,200,60), -1)
                    cv2.circle(frame, rapt, 8, (255,200,60), -1)

                    cv2.putText(frame, f"Reps: {rep_count}",
                                (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255,200,60), 3)
                    phase = "OPEN" if jj_state == "open" else "CLOSED"
                    cv2.putText(frame, phase,
                                (12, 68), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (220,220,220), 2)

                    with state.lock:
                        state.jj_state  = jj_state
                        state.rep_count = rep_count

                else:  # plank (default)
                    issues, angle_val, status_col, side, sh, hip, ank = analyse_plank(lms, w, h)
                    gf = (len(issues) == 0) and not standing

                    draw_skeleton(frame, lms, w, h, status_col)
                    cv2.line(frame, sh, ank, status_col, 3)
                    cv2.circle(frame, hip, 10, status_col,   -1)
                    cv2.circle(frame, sh,   8, (255,200,60), -1)
                    cv2.circle(frame, ank,  8, (255,200,60), -1)

                # ── Inactivity timeout for rep exercises ─────────────────────────
                REP_INACTIVE_SECS = 10.0
                if exercise in ('squat', 'pushup', 'mountain_climber', 'jumping_jack'):
                    if rep_count > prev_rep_count:
                        last_rep_time  = now   # start inactivity clock after first rep
                        prev_rep_count = rep_count
                    elif last_rep_time is not None and rep_count > 0:
                        inactive_for = now - last_rep_time
                        if inactive_for >= REP_INACTIVE_SECS - 5:
                            countdown = max(0, int(REP_INACTIVE_SECS - inactive_for) + 1)
                        if inactive_for >= REP_INACTIVE_SECS:
                            save_session(best_hold, name, rep_count, exercise)
                            audio.play("done", priority=True)
                            with state.lock:
                                state.running   = False
                                state.status    = "ended"
                                state.rep_count = rep_count
                            rep_count=0; prev_rep_count=0; last_rep_time=None
                            mc_state={'left':'down','right':'down'}
                            time.sleep(1)
                            continue

                # ── Standing detection (plank only — squat handles its own end) ───
                if standing and exercise == 'plank':
                    while not audio._q.empty():
                        try: audio._q.get_nowait()
                        except: break
                    if not was_standing:
                        standing_since = now
                        audio._q.put("paused")
                    if standing_since:
                        stood = now - standing_since
                        if stood >= STAND_TO_END - 5:
                            countdown = max(0, int(STAND_TO_END - stood) + 1)
                        if stood >= STAND_TO_END:
                            if best_hold > 0 or rep_count > 0:
                                save_session(best_hold, name, rep_count, exercise)
                            audio.play("done", priority=True)
                            with state.lock:
                                state.running    = False
                                state.status     = "ended"
                                state.best_hold  = best_hold
                                state.hold_time  = hold_time
                                state.rep_count  = rep_count
                            hold_time=0; good_form_start=None
                            was_good_form=False; was_standing=False
                            standing_since=None; last_milestone=0
                            time.sleep(1)
                            continue
                else:
                    standing_since = None

                # For squat, never trigger standing-end
                if exercise == 'squat':
                    standing_since = None

                # ── Timer (plank hold) ────────────────────────────────────
                if gf and exercise == 'plank':
                    if good_form_start is None: good_form_start = now
                    if now - good_form_start >= GOOD_FORM_BUFFER:
                        hold_time += 1/30
                else:
                    good_form_start = None
                best_hold = max(best_hold, hold_time)

                # ── Audio events ──────────────────────────────────────────
                if gf and not was_good_form:
                    audio.play("good")

                if gf and exercise == 'plank':
                    ms = (int(hold_time)//MILESTONE_EVERY)*MILESTONE_EVERY
                    if ms > 0 and ms != last_milestone:
                        last_milestone = ms
                        avail=[5,10,15,20,25,30,45,60,90,120]
                        audio.play(min(avail,key=lambda x:abs(x-ms)), priority=True)

                if not gf and issues and not standing:
                    txt, wkey = issues[0]
                    if now - last_issue_t.get(wkey, 0) > ISSUE_COOLDOWN:
                        audio.play(wkey, priority=True)
                        last_issue_t[wkey] = now

                # ── Squat audio events ────────────────────────────────────
                if exercise == 'squat':
                    # Announce each rep (1-9) then milestones at 5,10,15...
                    prev_reps = getattr(audio, '_prev_reps', 0)
                    if rep_count != prev_reps and rep_count > 0:
                        audio._prev_reps = rep_count
                        # Milestone at multiples of 5
                        sq_milestones = [5,10,15,20,25,30,40,50]
                        if rep_count in sq_milestones:
                            audio.play(f"sq_{rep_count}", priority=True)
                        elif rep_count < 10 and rep_count not in [5]:
                            # Count individual reps 1-9
                            audio.play(f"sq_rep{rep_count}", priority=True)

                    # Squat form warnings
                    if not gf and issues and not standing:
                        txt, wkey = issues[0]
                        sq_warn_map = {
                            "warn_hips_low": "sq_lean",
                            "warn_hips_high": "sq_lean",
                            "sq_lean": "sq_lean",
                            "sq_depth": "sq_depth",
                        }
                        sq_key = sq_warn_map.get(wkey, wkey)
                        if now - last_issue_t.get(sq_key, 0) > ISSUE_COOLDOWN:
                            audio.play(sq_key, priority=False)
                            last_issue_t[sq_key] = now

                was_good_form = gf
                was_standing  = standing
                feedback = issues[0][0] if issues and not standing else (
                    "Standing — session paused" if standing else
                    "Perfect form — hold steady!" if gf else ""
                )

            else:
                if now - last_seen > 3.0:
                    hold_time=0; good_form_start=None
                    was_good_form=False; was_standing=False
                    last_milestone=0; standing_since=None
                feedback = "No pose detected — stand sideways to camera"

            # Encode frame
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with state.lock:
                state.latest_frame = buf.tobytes()
                state.hold_time    = hold_time
                state.best_hold    = best_hold
                state.feedback     = feedback
                state.angle        = angle_val
                state.good_form    = gf
                state.standing     = standing
                state.countdown    = countdown
                state.status       = (
                    "standing" if standing else
                    "good"     if gf else
                    "active"
                )

    cap.release()

# ─── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=resource_path("static"))

def _make_placeholder_frame():
    """Dark grey frame with 'Camera loading...' text shown before webcam is ready."""
    import numpy as np
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, "Camera initialising...",
                (140, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120,120,120), 2)
    cv2.putText(img, "Please wait",
                (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 1)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

_PLACEHOLDER_FRAME = None

def generate_frames():
    global _PLACEHOLDER_FRAME
    if _PLACEHOLDER_FRAME is None:
        _PLACEHOLDER_FRAME = _make_placeholder_frame()
    while True:
        with state.lock:
            frame = state.latest_frame
        f = frame if frame else _PLACEHOLDER_FRAME
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')
        time.sleep(0.033)  # ~30fps

@app.route('/')
def index():
    return send_from_directory(resource_path('static'), 'index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def stats():
    with state.lock:
        return jsonify({
            "running":        state.running,
            "status":         state.status,
            "hold_time":      round(state.hold_time, 1),
            "best_hold":      round(state.best_hold, 1),
            "hold_fmt":       fmt_time(state.hold_time),
            "best_fmt":       fmt_time(state.best_hold),
            "feedback":       state.feedback,
            "angle":          round(state.angle, 1),
            "good_form":      state.good_form,
            "standing":       state.standing,
            "countdown":      state.countdown,
            "prep_countdown": state.prep_countdown,
            "name":           state.player_name,
            "exercise":       state.exercise,
            "rep_count":      state.rep_count,
            "jj_state":       state.jj_state,
        })

def _final_countdown():
    """Play 'Starting now!' then 5-4-3-2-1-Go then start tracking."""
    with state.lock:
        if state.status not in ('preparing', 'final_countdown'):
            return
        state.status = 'final_countdown'
        state.prep_countdown = 0

    # Play "Starting now!" and wait for it to finish before counting
    audio.play("starting_now", priority=True)
    time.sleep(1.8)   # duration of "Starting now!" wav

    sequence = [
        ("prep_5", 1),
        ("prep_4", 1),
        ("prep_3", 1),
        ("prep_2", 1),
        ("prep_1", 1),
        ("prep_go", 0.8),
    ]
    for key, delay in sequence:
        with state.lock:
            if state.status != 'final_countdown':
                return   # session cancelled mid-countdown
            state.prep_countdown = int(key.split('_')[1]) if key != 'prep_go' else 0
        audio.play(key, priority=True)
        if delay:
            time.sleep(delay)

    # Check again before starting — user may have cancelled
    with state.lock:
        if state.status != 'final_countdown':
            return
        state.running        = True
        state.status         = "active"
        state.prep_countdown = 0
        state.reset_signal   = True

def _begin_active_session():
    """Transition from preparing → final_countdown → active."""
    with state.lock:
        if state.status not in ('preparing',):
            return   # already started, in countdown, or cancelled
        state.status = 'final_countdown'
    t = threading.Thread(target=_final_countdown, daemon=True)
    t.start()

def _prep_timer(name, exercise):
    """Background thread: counts down PREP_COUNTDOWN_SECS then fires final countdown.
    Uses wall-clock anchoring so display and audio stay in sync."""
    start = time.time()
    deadline = start + PREP_COUNTDOWN_SECS

    # Announce at start — don't sleep, audio is async queued
    audio.play("get_ready", priority=True)
    audio.play("say_ready", priority=False)

    announced = set()   # track which milestones we've already announced

    while True:
        now = time.time()
        remaining = max(0, int(deadline - now))

        with state.lock:
            if state.status != 'preparing':
                return   # user triggered Ready early
            state.prep_countdown = remaining

        # Announce at milestones — only once each, keyed to display value
        if remaining == 15 and 15 not in announced:
            announced.add(15)
            audio.play("prep_ann_15", priority=False)
        elif remaining == 10 and 10 not in announced:
            announced.add(10)
            audio.play("prep_ann_10", priority=False)
        elif remaining == 5 and 5 not in announced:
            announced.add(5)
            audio.play("prep_ann_5", priority=False)

        if remaining == 0:
            break

        time.sleep(0.1)   # tight loop for accurate display, low CPU

    # Time's up — start final countdown
    with state.lock:
        if state.status != 'preparing':
            return
    _begin_active_session()

@app.route('/api/start', methods=['POST'])
def start():
    data = request.get_json() or {}
    name     = data.get('name','').strip() or 'Anonymous'
    exercise = data.get('exercise','plank')
    audio.flush()   # kill any leftover audio from previous session
    with state.lock:
        state.player_name    = name
        state.exercise       = exercise
        state.running        = False          # not tracking yet
        state.status         = "preparing"    # countdown phase
        state.prep_countdown = PREP_COUNTDOWN_SECS
        state.hold_time      = 0.0
        state.best_hold      = 0.0
        state.rep_count      = 0
        state.squat_state    = 'up'
        state.pu_state       = 'up'
        state.mc_state       = {'left':'down','right':'down'}
        state.jj_state       = 'closed'
        state.reset_signal   = True    # reset camera thread local vars immediately
    # Launch prep countdown in background
    t = threading.Thread(target=_prep_timer, args=(name, exercise), daemon=True)
    t.start()
    return jsonify({"ok": True, "name": name, "prep_countdown": PREP_COUNTDOWN_SECS})

@app.route('/api/ready', methods=['POST'])
def ready():
    """User said 'Ready' or clicked Start Now — skip prep countdown, begin final 5s countdown."""
    with state.lock:
        already_counting = state.status == 'final_countdown'
    if not already_counting:
        _begin_active_session()
    return jsonify({"ok": True})

@app.route('/api/reset', methods=['POST'])
def reset_session():
    with state.lock:
        state.hold_time = 0.0
    audio.flush()
    audio.play("reset")
    return jsonify({"ok": True})

@app.route('/api/end', methods=['POST'])
def end_session():
    with state.lock:
        state.running = False
        state.status  = "ended"
        bh       = state.best_hold
        name     = state.player_name
        reps     = state.rep_count
        exercise = state.exercise
    audio.flush()   # kill any queued prep/countdown audio immediately
    if bh > 0 or reps > 0:
        save_session(bh, name, reps, exercise)
    audio.play("done", priority=True)
    return jsonify({"ok": True, "best_hold": bh, "best_fmt": fmt_time(bh), "reps": reps})


@app.route('/api/quit', methods=['POST'])
def quit_app():
    """Gracefully shut down the server."""
    import threading
    def shutdown():
        import time; time.sleep(0.5)
        import os, signal
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=shutdown, daemon=True).start()
    return jsonify({"status": "shutting_down"})

# ─── HEARTBEAT / AUTO-SHUTDOWN ────────────────────────────────────────────────
_last_heartbeat = time.time()
_HEARTBEAT_TIMEOUT = 8   # seconds without a ping before auto-shutdown

@app.route('/api/heartbeat', methods=['POST'])
def heartbeat():
    global _last_heartbeat
    _last_heartbeat = time.time()
    return jsonify({"ok": True})

def _heartbeat_watchdog():
    """Shutdown if browser hasn't pinged for _HEARTBEAT_TIMEOUT seconds."""
    global _last_heartbeat
    # Reset heartbeat timestamp to now so the grace period starts
    # from when the app is actually ready, not from process start
    _last_heartbeat = time.time()
    # Give user 60s to open the browser after the app prints its URL
    time.sleep(60)
    while True:
        time.sleep(2)
        if time.time() - _last_heartbeat > _HEARTBEAT_TIMEOUT:
            print("No heartbeat — browser closed. Shutting down.")
            import os, signal
            os.kill(os.getpid(), signal.SIGTERM)

@app.route('/api/summary')
def summary():
    """Return weekly stats from the session log."""
    import csv, os
    from datetime import datetime, timedelta
    rows = []
    if os.path.exists('plank_log.csv'):
        with open('plank_log.csv', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    # Last 7 days
    week_ago = datetime.now() - timedelta(days=7)
    recent = []
    for r in rows:
        try:
            dt = datetime.strptime(r['date'] + ' ' + r['time'], '%Y-%m-%d %H:%M')
            if dt >= week_ago:
                recent.append(r)
        except:
            pass

    # Total sessions this week
    total_sessions = len(recent)

    # Best per exercise this week
    best = {}
    for r in recent:
        ex = r.get('exercise', 'plank')
        if ex in ['squat','pushup','mountain_climber','jumping_jack']:
            val = int(r.get('reps', 0) or 0)
        else:
            val = int(r.get('best_hold_sec', 0) or 0)
        if ex not in best or val > best[ex]:
            best[ex] = val

    # Today's sessions
    today = datetime.now().strftime('%Y-%m-%d')
    today_sessions = [r for r in recent if r.get('date') == today]

    return jsonify({
        "total_sessions": total_sessions,
        "today_sessions": len(today_sessions),
        "best": best,
        "all_rows": rows[-20:],  # last 20 sessions
    })

@app.route('/api/log')
def log():
    resp = jsonify(load_log())
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    return resp

# ─── ADMIN CONFIG ─────────────────────────────────────────────────────────────
import json
import hashlib

CONFIG_FILE = "notgym_config.json"
PROFILES_FILE = "notgym_profiles.json"

def _hash_pin(pin):
    return hashlib.sha256(pin.encode()).hexdigest()

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    # Default config
    return {"pin_hash": _hash_pin("1234")}

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return []

def save_profiles(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)

def verify_pin(pin):
    cfg = load_config()
    return cfg.get("pin_hash") == _hash_pin(pin)

# ─── ADMIN ROUTES ─────────────────────────────────────────────────────────────

@app.route('/admin')
def admin():
    return send_from_directory(resource_path('static'), 'admin.html')

@app.route('/api/admin/verify', methods=['POST'])
def admin_verify():
    data = request.get_json() or {}
    if verify_pin(str(data.get('pin', ''))):
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Invalid PIN"}), 401

@app.route('/api/admin/change_pin', methods=['POST'])
def admin_change_pin():
    data = request.get_json() or {}
    if not verify_pin(str(data.get('current_pin', ''))):
        return jsonify({"ok": False, "error": "Invalid current PIN"}), 401
    new_pin = str(data.get('new_pin', '')).strip()
    if len(new_pin) < 4:
        return jsonify({"ok": False, "error": "PIN must be at least 4 digits"}), 400
    cfg = load_config()
    cfg['pin_hash'] = _hash_pin(new_pin)
    save_config(cfg)
    return jsonify({"ok": True})

@app.route('/api/admin/delete_history', methods=['POST'])
def admin_delete_history():
    data = request.get_json() or {}
    if not verify_pin(str(data.get('pin', ''))):
        return jsonify({"ok": False, "error": "Invalid PIN"}), 401
    if data.get('confirm') != 'DELETE':
        return jsonify({"ok": False, "error": "Type DELETE to confirm"}), 400
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    return jsonify({"ok": True})

@app.route('/api/admin/export_csv')
def admin_export_csv():
    pin = request.args.get('pin', '')
    if not verify_pin(pin):
        return jsonify({"ok": False, "error": "Invalid PIN"}), 401
    if not os.path.exists(LOG_FILE):
        return jsonify({"ok": False, "error": "No log file found"}), 404
    from flask import send_file
    return send_file(LOG_FILE, as_attachment=True, download_name="notgym_history.csv")

@app.route('/api/admin/profiles', methods=['GET'])
def admin_get_profiles():
    pin = request.args.get('pin', '')
    if not verify_pin(pin):
        return jsonify({"ok": False, "error": "Invalid PIN"}), 401
    return jsonify({"ok": True, "profiles": load_profiles()})

@app.route('/api/admin/profiles', methods=['POST'])
def admin_save_profiles():
    data = request.get_json() or {}
    if not verify_pin(str(data.get('pin', ''))):
        return jsonify({"ok": False, "error": "Invalid PIN"}), 401
    profiles = data.get('profiles', [])
    save_profiles(profiles)
    return jsonify({"ok": True})

@app.route('/api/admin/leaderboard')
def admin_leaderboard():
    pin = request.args.get('pin', '')
    if not verify_pin(pin):
        return jsonify({"ok": False, "error": "Invalid PIN"}), 401

    period = request.args.get('period', 'all')   # week, month, all
    group  = request.args.get('group', 'all')    # group name or 'all'

    rows = load_log()
    profiles = load_profiles()

    # Build group lookup
    group_map = {p['name']: p.get('group', '') for p in profiles}

    # Filter by period
    from datetime import timedelta
    now = datetime.now()
    if period == 'week':
        cutoff = now - timedelta(days=7)
    elif period == 'month':
        cutoff = now - timedelta(days=30)
    else:
        cutoff = None

    filtered = []
    for r in rows:
        if cutoff:
            try:
                dt = datetime.strptime(r['date'] + ' ' + r['time'], '%Y-%m-%d %H:%M')
                if dt < cutoff:
                    continue
            except:
                pass
        # Filter by group
        if group != 'all':
            if group_map.get(r.get('name', ''), '') != group:
                continue
        filtered.append(r)

    # Aggregate per student
    stats = {}
    for r in filtered:
        name = r.get('name', 'Anonymous')
        if name not in stats:
            stats[name] = {
                'name': name,
                'group': group_map.get(name, ''),
                'sessions': 0,
                'best_plank': 0,
                'best_squats': 0,
                'best_pushups': 0,
                'best_mountain_climber': 0,
                'best_jumping_jack': 0,
            }
        s = stats[name]
        s['sessions'] += 1
        ex = r.get('exercise', 'plank')
        if ex == 'plank':
            val = float(r.get('best_hold_sec', 0) or 0)
            s['best_plank'] = max(s['best_plank'], val)
        elif ex == 'squat':
            val = int(r.get('reps', 0) or 0)
            s['best_squats'] = max(s['best_squats'], val)
        elif ex == 'pushup':
            val = int(r.get('reps', 0) or 0)
            s['best_pushups'] = max(s['best_pushups'], val)
        elif ex == 'mountain_climber':
            val = int(r.get('reps', 0) or 0)
            s['best_mountain_climber'] = max(s['best_mountain_climber'], val)
        elif ex == 'jumping_jack':
            val = int(r.get('reps', 0) or 0)
            s['best_jumping_jack'] = max(s['best_jumping_jack'], val)

    # Get all groups for filter dropdown
    all_groups = sorted(set(p.get('group', '') for p in profiles if p.get('group')))

    return jsonify({
        "ok": True,
        "leaderboard": list(stats.values()),
        "groups": all_groups,
        "period": period,
        "group": group,
    })

# ─── PROFILES API (for main screen dropdown) ──────────────────────────────────

@app.route('/api/profiles')
def get_profiles_public():
    """Public endpoint — returns just names and groups for the student dropdown."""
    profiles = load_profiles()
    return jsonify([{"name": p['name'], "group": p.get('group', '')} for p in profiles])

if __name__ == '__main__':
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    v = threading.Thread(target=voice_listener_loop, daemon=True)
    v.start()
    w = threading.Thread(target=_heartbeat_watchdog, daemon=True)
    w.start()
    _last_heartbeat = time.time()   # reset so 60s grace starts from here
    print("NotGym running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
