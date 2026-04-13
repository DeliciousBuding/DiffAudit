param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,

    [Parameter(Mandatory = $true)]
    [string]$AssetsRoot,

    [string]$RuntimeWorkspace = "D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-runtime-mainline-20260408-cifar10-1k-3shadow"
)

$ErrorActionPreference = "Stop"

while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 15
}

powershell -ExecutionPolicy Bypass -File "D:\Code\DiffAudit\Research\scripts\launch_gsa_training_sequence.ps1" `
    -AssetsRoot $AssetsRoot `
    -ShadowOnly `
    -RuntimeWorkspace $RuntimeWorkspace
