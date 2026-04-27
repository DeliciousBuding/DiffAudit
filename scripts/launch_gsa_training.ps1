param(
    [Parameter(Mandatory = $true)]
    [string]$TrainDataDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [string]$PythonExe = "python",
    [string]$TrainScript = "",
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

if ($TrainScript -eq "") {
    $TrainScript = Join-Path (Split-Path -Parent $PSScriptRoot) "workspaces\white-box\external\GSA\DDPM\train_unconditional.py"
}

if (Test-Path -LiteralPath $PythonExe) {
    $resolvedPythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
} else {
    $pythonCommand = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python executable not found: $PythonExe"
    }
    $resolvedPythonExe = $pythonCommand.Source
}

if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "Train script not found: $TrainScript"
}

$resolvedTrainScript = (Resolve-Path -LiteralPath $TrainScript).Path

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$stdout = Join-Path $OutputDir "train.stdout.log"
$stderr = Join-Path $OutputDir "train.stderr.log"
if (Test-Path $stdout) { Remove-Item -LiteralPath $stdout -Force }
if (Test-Path $stderr) { Remove-Item -LiteralPath $stderr -Force }

$args = @(
    $resolvedTrainScript,
    "--train_data_dir", $TrainDataDir,
    "--resolution", "$Resolution",
    "--output_dir", $OutputDir,
    "--train_batch_size", "$TrainBatchSize",
    "--num_epochs", "$NumEpochs",
    "--gradient_accumulation_steps", "$GradientAccumulationSteps",
    "--learning_rate", $LearningRate,
    "--lr_warmup_steps", "$LrWarmupSteps",
    "--mixed_precision", $MixedPrecision,
    "--save_images_epochs", "$SaveImagesEpochs",
    "--save_model_epochs", "$SaveModelEpochs",
    "--prediction_type", $PredictionType
)

$process = Start-Process `
    -FilePath $resolvedPythonExe `
    -ArgumentList $args `
    -WorkingDirectory (Split-Path -Parent $resolvedTrainScript) `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

[pscustomobject]@{
    pid = $process.Id
    stdout = $stdout
    stderr = $stderr
    output_dir = $OutputDir
    train_data_dir = $TrainDataDir
} | ConvertTo-Json -Compress
