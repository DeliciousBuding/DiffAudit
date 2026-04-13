param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,

    [Parameter(Mandatory = $true)]
    [string]$AssetsRoot,

    [string]$WorkspaceRoot = $env:DIFFAUDIT_WORKSPACE_ROOT,
    [string]$ResearchRoot = "",
    [string]$RunSuffix = "strong-v2",
    [string]$PythonExe = $env:DIFFAUDIT_RESEARCH_PYTHON,
    [string]$DpdmRoot = "",
    [string]$ConfigPath = "",
    [int]$Epochs = 3,
    [int]$SigmaNoiseSamples = 2,
    [int]$BasePort = 6040
)

$ErrorActionPreference = "Stop"

while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 15
}

$launchShadowScript = Join-Path $PSScriptRoot "launch_dpdm_shadow_sequence.ps1"

$shadowArgs = @(
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $launchShadowScript,
    "-AssetsRoot",
    $AssetsRoot,
    "-RunSuffix",
    $RunSuffix,
    "-Epochs",
    $Epochs,
    "-SigmaNoiseSamples",
    $SigmaNoiseSamples,
    "-BasePort",
    $BasePort
)
if (-not [string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $shadowArgs += @("-WorkspaceRoot", $WorkspaceRoot)
}
if (-not [string]::IsNullOrWhiteSpace($ResearchRoot)) {
    $shadowArgs += @("-ResearchRoot", $ResearchRoot)
}
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    $shadowArgs += @("-PythonExe", $PythonExe)
}
if (-not [string]::IsNullOrWhiteSpace($DpdmRoot)) {
    $shadowArgs += @("-DpdmRoot", $DpdmRoot)
}
if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $shadowArgs += @("-ConfigPath", $ConfigPath)
}

powershell @shadowArgs
