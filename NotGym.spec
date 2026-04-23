# -*- mode: python ; coding: utf-8 -*-
import glob
import os

wav_files = [(f, '.') for f in glob.glob(r'C:\Users\graja\Downloads\NotGym\*.wav')]

# Vosk model folder
vosk_model_dir = r'C:\Users\graja\Downloads\NotGym\vosk-model-small-en-us'
vosk_datas = [(vosk_model_dir, 'vosk-model-small-en-us')] if os.path.isdir(vosk_model_dir) else []

# Vosk package folder (for DLL)
vosk_pkg_dir = r'C:\Users\graja\AppData\Local\Programs\Python\Python311\Lib\site-packages\vosk'
vosk_pkg_datas = [(vosk_pkg_dir, 'vosk')] if os.path.isdir(vosk_pkg_dir) else []

a = Analysis(
    ['activeai.py'],
    pathex=[],
    binaries=[],
    datas=[
        (r'C:\Users\graja\AppData\Local\Programs\Python\Python311\Lib\site-packages\mediapipe', 'mediapipe'),
        ('static', 'static'),
        ('pose_landmarker_lite.task', '.'),
    ] + wav_files + vosk_datas + vosk_pkg_datas,
    hiddenimports=[
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.vision',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'mediapipe.python.solutions.pose',
        'mediapipe.python.solutions.drawing_utils',
        'vosk',
        'sounddevice',
        'cffi',
        '_cffi_backend',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='NotGym',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
