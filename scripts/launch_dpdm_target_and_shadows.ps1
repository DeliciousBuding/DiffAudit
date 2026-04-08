param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,

    [Parameter(Mandatory = $true)]
    [string]$AssetsRoot,

    [Parameter(Mandatory = $true)]
    [string]$Workdir,

    [string]$RunSuffix = "strong-v2",
    [string]$PythonExe = "C:\Users\Ding\miniforge3\envs\diffaudit-research\python.exe",
    [string]$DpdmRoot = "D:\Code\DiffAudit\Project\external\DPDM",
    [string]$ConfigPath = "D:\Code\DiffAudit\Project\external\DPDM\configs\cifar10_32\train_eps_10.0.yaml",
    [int]$BatchSize = 64,
    [int]$Epochs = 3,
    [int]$SigmaNoiseSamples = 2,
    [string]$DeviceBackend = "gloo",
    [int]$MasterPort = 6035,
    [int]$BasePort = 6040
)

$ErrorActionPreference = "Stop"

$launchTargetScript = "D:\Code\DiffAudit\Project\scripts\launch_dpdm_training.ps1"
$launchShadowScript = "D:\Code\DiffAudit\Project\scripts\launch_dpdm_shadow_sequence_after_pid.ps1"

$targetLaunchJson = powershell -ExecutionPolicy Bypass -File $launchTargetScript `
    -DataPath $DataPath `
    -Workdir $Workdir `
    -PythonExe $PythonExe `
    -DpdmRoot $DpdmRoot `
    -ConfigPath $ConfigPath `
    -BatchSize $BatchSize `
    -Epochs $Epochs `
    -SigmaNoiseSamples $SigmaNoiseSamples `
    -DeviceBackend $DeviceBackend `
    -MasterPort $MasterPort

$targetLaunch = $targetLaunchJson | ConvertFrom-Json
$targetPid = [int]$targetLaunch.pid

$shadowWatcher = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $launchShadowScript,
        "-WaitPid",
        $targetPid,
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
    ) `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    target_pid = $targetPid
    target_stdout = $targetLaunch.stdout
    target_stderr = $targetLaunch.stderr
    target_workdir = $targetLaunch.workdir
    shadow_watcher_pid = $shadowWatcher.Id
    run_suffix = $RunSuffix
} | ConvertTo-Json -Compress
