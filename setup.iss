#define MyAppName "SpikeHound"
#define MyAppVersion "2.0"
#define MyAppPublisher "Gus K. Lott III"
#define MyAppURL "https://github.com/guslott/spikehound"
#define MyAppExeName "SpikeHound.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside Inno Setup)
AppId={{C626C8D6-2771-4471-8507-6A6F57635079}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; Only allow running on 64-bit Windows (standard for scientific apps)
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
; LicenseFile=LICENSE  <-- Uncomment if you want the user to agree to the license
OutputDir=dist
OutputBaseFilename=SpikeHound_Windows_Installer
; This uses the icon you committed to the media folder
SetupIconFile=media\SpikeHound.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; IMPORTANT: This grabs the output from PyInstaller's folder mode
Source: "dist\SpikeHound\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent