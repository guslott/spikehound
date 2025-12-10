# -*- mode: python ; coding: utf-8 -*-
#Usage: pyinstaller --clean --noconfirm SpikeHound.spec

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("media/mph_cornell_splash.png", "media"),
        # Add any other assets you want inside the app bundle, e.g.:
        # ("docs/SpikeHound-v1p2b-Manual.pdf", "docs"),
    ],
    hiddenimports=[
        "daq.simulated_source",
        "daq.backyard_brains_source",
        "daq.soundcard_source",
        "daq.file_source",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],                      # Don't include binaries/datas here - COLLECT does that
    name="SpikeHound",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # GUI app, no terminal window
    exclude_binaries=True,   # Required for one-folder mode with COLLECT
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,               # UPX often causes issues on macOS
    upx_exclude=[],
    name="SpikeHound",
)

app = BUNDLE(
    coll,                   # Use COLLECT output for proper .app bundle
    name="SpikeHound.app",
    icon="media/SpikeHound.icns",     # set to None if you don't have an icon yet
    bundle_identifier="org.mphschool.spikehound",
)