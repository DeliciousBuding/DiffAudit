param(
    [Parameter(Mandatory = $true)]
    [string]$AssetsRoot,

    [string[]]$Splits = @("target-member", "shadow-01-member", "shadow-02-member", "shadow-03-member"),
    [switch]$ShadowOnly,
    [string]$RuntimeWorkspace = "",
    [string]$RepoRoot = "D:\Code\DiffAudit\Research\workspaces\white-box\external\GSA",
    [string]$PythonExe = "C:\Users\Ding\miniforge3\envs\diffaudit-research\python.exe",
    [string]$TrainScript = "D:\Code\DiffAudit\Research\workspaces\white-box\external\GSA\DDPM\train_unconditional.py",
    [int]$Resolution = 32,
    [int]$TrainBatchSize = 32,
    [int]$NumEpochs = 200,
    [int]$GradientAccumulationSteps = 1,
    [string]$LearningRate = "1e-4",
    [int]$LrWarmupSteps = 500,
    [string]$MixedPrecision = "no",
    [int]$SaveImagesEpochs = 100000,
    [int]$SaveModelEpochs = 50,
    [string]$PredictionType = "epsilon"
)

$ErrorActionPreference = "Stop"

$launchScript = "D:\Code\DiffAudit\Research\scripts\launch_gsa_training.ps1"

if ($ShadowOnly) {
    $Splits = @("shadow-01-member", "shadow-02-member", "shadow-03-member")
}

foreach ($split in $Splits) {
    $splitPrefix = $split -replace "-member$", ""
    $trainDataDir = Join-Path $AssetsRoot "datasets\$split"
    $outputDir = Join-Path $AssetsRoot "checkpoints\$splitPrefix"

    Write-Host "Launching $split -> $outputDir"
    $launchResultJson = & $launchScript `
        -TrainDataDir $trainDataDir `
        -OutputDir $outputDir `
        -PythonExe $PythonExe `
        -TrainScript $TrainScript `
        -Resolution $Resolution `
        -TrainBatchSize $TrainBatchSize `
        -NumEpochs $NumEpochs `
        -GradientAccumulationSteps $GradientAccumulationSteps `
        -LearningRate $LearningRate `
        -LrWarmupSteps $LrWarmupSteps `
        -MixedPrecision $MixedPrecision `
        -SaveImagesEpochs $SaveImagesEpochs `
        -SaveModelEpochs $SaveModelEpochs `
        -PredictionType $PredictionType

    $launchResult = $launchResultJson | ConvertFrom-Json
    $childPid = [int]$launchResult.pid
    Write-Host "Spawned PID $childPid"

    while (Get-Process -Id $childPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 15
    }

    Write-Host "Completed $split"
}

if ($RuntimeWorkspace -ne "") {
    Write-Host "Launching paper-aligned GSA runtime mainline"
    & $PythonExe -m diffaudit run-gsa-runtime-mainline `
        --workspace $RuntimeWorkspace `
        --repo-root $RepoRoot `
        --assets-root $AssetsRoot `
        --device cuda `
        --paper-aligned `
        --provenance-status workspace-verified
}
