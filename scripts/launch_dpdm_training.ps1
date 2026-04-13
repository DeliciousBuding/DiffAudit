param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,

    [Parameter(Mandatory = $true)]
    [string]$Workdir,

    [string]$WorkspaceRoot = $env:DIFFAUDIT_WORKSPACE_ROOT,
    [string]$ResearchRoot = "",
    [string]$PythonExe = $env:DIFFAUDIT_RESEARCH_PYTHON,
    [string]$DpdmRoot = "",
    [string]$ConfigPath = "",
    [int]$BatchSize = 64,
    [int]$Epochs = 1,
    [int]$SigmaNoiseSamples = 1,
    [string]$DeviceBackend = "gloo",
    [int]$MasterPort = 6030
)

$ErrorActionPreference = "Stop"

function Resolve-ExistingPath([string]$PathValue) {
    return (Resolve-Path -LiteralPath $PathValue).Path
}

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

$resolvedResearchRoot = Resolve-ExistingPath $ResearchRoot
if ([string]::IsNullOrWhiteSpace($DpdmRoot)) {
    $DpdmRoot = Join-Path $resolvedResearchRoot "external/DPDM"
}
$resolvedDpdmRoot = Resolve-ExistingPath $DpdmRoot

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $resolvedDpdmRoot "configs/cifar10_32/train_eps_10.0.yaml"
}

$resolvedPythonExe = Resolve-ExecutableCandidate $PythonExe
$resolvedConfigPath = Resolve-ExistingPath $ConfigPath
$resolvedDataPath = Resolve-ExistingPath $DataPath

$stdout = Join-Path $resolvedDpdmRoot "$Workdir.stdout.log"
$stderr = Join-Path $resolvedDpdmRoot "$Workdir.stderr.log"
if (Test-Path $stdout) { Remove-Item -LiteralPath $stdout -Force }
if (Test-Path $stderr) { Remove-Item -LiteralPath $stderr -Force }

$args = @(
    (Join-Path $resolvedDpdmRoot "main.py"),
    "--mode", "train",
    "--workdir", $Workdir,
    "--config", $resolvedConfigPath,
    "--root_folder", $resolvedDpdmRoot,
    "--",
    "setup.n_gpus_per_node=1",
    "setup.omp_n_threads=8",
    "setup.backend=$DeviceBackend",
    "setup.master_address=127.0.0.1",
    "setup.master_port=$MasterPort",
    "data.path=$resolvedDataPath",
    "data.dataset_params.use_labels=true",
    "data.dataset_params.xflip=false",
    "train.batch_size=$BatchSize",
    "train.n_epochs=$Epochs",
    "train.log_freq=10",
    "train.snapshot_freq=100000",
    "train.save_freq=100000",
    "train.fid_freq=100000",
    "train.fid_threshold=100000",
    "train.save_threshold=100000",
    "train.snapshot_threshold=100000",
    "sampler.snapshot_batch_size=8",
    "sampler.fid_batch_size=8",
    "dp.max_physical_batch_size=64",
    "loss.n_classes=10",
    "loss.n_noise_samples=$SigmaNoiseSamples"
)

$process = Start-Process `
    -FilePath $resolvedPythonExe `
    -ArgumentList $args `
    -WorkingDirectory $resolvedDpdmRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

[pscustomobject]@{
    pid = $process.Id
    stdout = $stdout
    stderr = $stderr
    workdir = $Workdir
    data_path = $resolvedDataPath
    research_root = $resolvedResearchRoot
} | ConvertTo-Json -Compress
