param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,

    [Parameter(Mandatory = $true)]
    [string]$AssetsRoot,

    [string]$RuntimeWorkspace = ""
)

$ErrorActionPreference = "Stop"

if ($RuntimeWorkspace -eq "") {
    $RuntimeWorkspace = Join-Path (Split-Path -Parent $PSScriptRoot) "workspaces\white-box\runs\gsa-runtime-mainline-20260408-cifar10-1k-3shadow"
}

while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 15
}

powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "launch_gsa_training_sequence.ps1") `
    -AssetsRoot $AssetsRoot `
    -ShadowOnly `
    -RuntimeWorkspace $RuntimeWorkspace
