$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "D:\Programs\VideoCaptioner\runtime\python.exe"
$Url = "http://localhost:8000/listening_chunking_shadowing_trainer_optimized.html"

$isListening = $false
try {
  $connection = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction Stop | Select-Object -First 1
  if ($connection) { $isListening = $true }
} catch {
  $isListening = $false
}

if (-not $isListening) {
  Start-Process -FilePath $Python -ArgumentList "whisper_alignment_server.py" -WorkingDirectory $AppDir -WindowStyle Hidden
  Start-Sleep -Seconds 2
}

Start-Process $Url
