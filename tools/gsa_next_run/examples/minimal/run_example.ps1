param(
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$toolRoot = Resolve-Path (Join-Path $here "..\\..")

$assetsRoot = Join-Path $here "assets_root"
$repoRoot = Join-Path $here "repo_root"
$configPath = Join-Path $here "config"
$outDir = Join-Path $here "out"

if (-not (Test-Path (Join-Path $repoRoot ".git"))) {
  git -C $repoRoot init | Out-Host
  git -C $repoRoot config user.email "you@example.com"
  git -C $repoRoot config user.name "You"
  "# minimal repo" | Set-Content -Encoding ascii (Join-Path $repoRoot "README.md")
  git -C $repoRoot add README.md | Out-Host
  git -C $repoRoot commit -m "init" | Out-Host
}

Remove-Item -Recurse -Force $outDir -ErrorAction SilentlyContinue

$args = @(
  "--assets-root", $assetsRoot,
  "--repo-root", $repoRoot,
  "--config", $configPath,
  "--out-dir", $outDir
)
if ($Strict) { $args += "--strict" }

& (Join-Path $toolRoot "run.ps1") @args

