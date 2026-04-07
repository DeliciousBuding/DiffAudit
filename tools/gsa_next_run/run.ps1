param(
  [Parameter(ValueFromRemainingArguments=$true)]
  [string[]]$Args
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$here\src"

python -m gsa_next_run @Args

