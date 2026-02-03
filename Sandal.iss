#define MyAppName "Sandal"
#define MyAppVersion "1.0"
#define MyAppPublisher "Your Company"
#define MyAppExeName "Sandal.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{12345678-1234-1234-1234-123456789012}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={pf}\Sandal
DefaultGroupName=Sandal
OutputBaseFilename=Sandal-Setup
Compression=lzma
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}";

[Files]
Source: "dist\Sandal\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Sandal"; Filename: "{app}\Sandal.exe"
Name: "{commondesktop}\Sandal"; Filename: "{app}\Sandal.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Sandal.exe"; Description: "Launch Sandal"; Flags: shellexec postinstall skipifsilent