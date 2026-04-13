param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,

    [Parameter(Mandatory = $true)]
    [string]$AssetsRoot,

    [Parameter(Mandatory = $true)]
    [string]$Workdir,

    [string]$WorkspaceRoot = $env:DIFFAUDIT_WORKSPACE_ROOT,
    [string]$ResearchRoot = "",
    [string]$RunSuffix = "strong-v2",
    [string]$PythonExe = $env:DIFFAUDIT_RESEARCH_PYTHON,
    [string]$DpdmRoot = "",
    [string]$ConfigPath = "",
    [int]$BatchSize = 64,
    [int]$Epochs = 3,
    [int]$SigmaNoiseSamples = 2,
    [string]$DeviceBackend = "gloo",
    [int]$MasterPort = 6035,
    [int]$BasePort = 6040
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ResearchRoot)) {
    if (-not [string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
        $ResearchRoot = Join-Path $WorkspaceRoot "Research"
    }
    else {
        $ResearchRoot = Join-Path $PSScriptRoot ".."
    }
}

$resolvedResearchRoot = (Resolve-Path -LiteralPath $ResearchRoot).Path
$resolvedAssetsRoot = (Resolve-Path -LiteralPath $AssetsRoot).Path
$launchTargetScript = Join-Path $PSScriptRoot "launch_dpdm_training.ps1"
$launchShadowScript = Join-Path $PSScriptRoot "launch_dpdm_shadow_sequence_after_pid.ps1"

$targetArgs = @(
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $launchTargetScript,
    "-DataPath",
    $DataPath,
    "-Workdir",
    $Workdir,
    "-ResearchRoot",
    $resolvedResearchRoot,
    "-BatchSize",
    $BatchSize,
    "-Epochs",
    $Epochs,
    "-SigmaNoiseSamples",
    $SigmaNoiseSamples,
    "-DeviceBackend",
    $DeviceBackend,
    "-MasterPort",
    $MasterPort
)
if (-not [string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $targetArgs += @("-WorkspaceRoot", $WorkspaceRoot)
}
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    $targetArgs += @("-PythonExe", $PythonExe)
}
if (-not [string]::IsNullOrWhiteSpace($DpdmRoot)) {
    $targetArgs += @("-DpdmRoot", $DpdmRoot)
}
if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $targetArgs += @("-ConfigPath", $ConfigPath)
}

$targetLaunchJson = powershell @targetArgs

$targetLaunch = $targetLaunchJson | ConvertFrom-Json
$targetPid = [int]$targetLaunch.pid

$shadowArgs = @(
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $launchShadowScript,
    "-WaitPid",
    $targetPid,
    "-AssetsRoot",
    $resolvedAssetsRoot,
    "-ResearchRoot",
    $resolvedResearchRoot,
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
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    $shadowArgs += @("-PythonExe", $PythonExe)
}
if (-not [string]::IsNullOrWhiteSpace($DpdmRoot)) {
    $shadowArgs += @("-DpdmRoot", $DpdmRoot)
}
if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $shadowArgs += @("-ConfigPath", $ConfigPath)
}

$shadowWatcher = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $shadowArgs `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    target_pid = $targetPid
    target_stdout = $targetLaunch.stdout
    target_stderr = $targetLaunch.stderr
    target_workdir = $targetLaunch.workdir
    shadow_watcher_pid = $shadowWatcher.Id
    run_suffix = $RunSuffix
    research_root = $resolvedResearchRoot
} | ConvertTo-Json -Compress
