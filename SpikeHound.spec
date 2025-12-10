import sys
import os

# usage: pyinstaller --clean --noconfirm SpikeHound.spec

block_cipher = None

# Platform-specific settings
if sys.platform == 'darwin':
    icon_file = os.path.join("media", "SpikeHound.icns")
    os_name = 'macOS'
else:
    icon_file = os.path.join("media", "SpikeHound.ico")
    os_name = 'Windows'

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        (os.path.join("media", "mph_cornell_splash.png"), "media"),
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
    [],
    exclude_binaries=True,
    name="SpikeHound",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SpikeHound",
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name="SpikeHound.app",
        icon=icon_file,
        bundle_identifier="org.mphschool.spikehound",
    )