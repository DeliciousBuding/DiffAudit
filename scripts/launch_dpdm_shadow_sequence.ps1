param(
    [Parameter(Mandatory = $true)]
    [string]$AssetsRoot,

    [string[]]$Shadows = @("shadow-01", "shadow-02", "shadow-03"),
    [string]$WorkspaceRoot = $env:DIFFAUDIT_WORKSPACE_ROOT,
    [string]$ResearchRoot = "",
    [string]$RunSuffix = "smoke-v1",
    [string]$PythonExe = $env:DIFFAUDIT_RESEARCH_PYTHON,
    [string]$DpdmRoot = "",
    [string]$ConfigPath = "",
    [int]$BatchSize = 64,
    [int]$Epochs = 1,
    [int]$SigmaNoiseSamples = 1,
    [string]$DeviceBackend = "gloo",
    [int]$BasePort = 6032
)

$ErrorActionPreference = "Stop"

function Resolve-ExecutableCandidate([string]$Candidate) {
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return "python"
    }
    if (Test-Path -LiteralPath $Candidate) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }
    return $Candidate
}

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
$resolvedPythonExe = Resolve-ExecutableCandidate $PythonExe
$writeLabelsScript = Join-Path $PSScriptRoot "write_imagefolder_labels.py"
$launchDpdmScript = Join-Path $PSScriptRoot "launch_dpdm_training.ps1"

for ($i = 0; $i -lt $Shadows.Length; $i++) {
    $shadowName = $Shadows[$i]
    $datasetDir = Join-Path $resolvedAssetsRoot "datasets\$shadowName-member"
    $workdir = "runs/dpdm-cifar10-$($shadowName.Replace('-', ''))-eps10-gpu-$RunSuffix"
    $masterPort = $BasePort + $i

    Write-Host "Preparing labels for $shadowName"
    & $resolvedPythonExe $writeLabelsScript --dataset-dir $datasetDir | Out-Host

    Write-Host "Launching DPDM $shadowName -> $workdir"
    $launchArgs = @(
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $launchDpdmScript,
        "-DataPath",
        $datasetDir,
        "-Workdir",
        $workdir,
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
        $masterPort
    )
    if (-not [string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
        $launchArgs += @("-WorkspaceRoot", $WorkspaceRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        $launchArgs += @("-PythonExe", $PythonExe)
    }
    if (-not [string]::IsNullOrWhiteSpace($DpdmRoot)) {
        $launchArgs += @("-DpdmRoot", $DpdmRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
        $launchArgs += @("-ConfigPath", $ConfigPath)
    }

    $launchJson = powershell @launchArgs

    $launch = $launchJson | ConvertFrom-Json
    $childPid = [int]$launch.pid
    Write-Host "Spawned PID $childPid"

    while (Get-Process -Id $childPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 15
    }

    Write-Host "Completed DPDM $shadowName"
}
