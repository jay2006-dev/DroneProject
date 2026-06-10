# One-time environment setup (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File setup.ps1            # core only
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Deep      # core + deep
param([switch]$Deep)

python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
if ($Deep) {
    & .\.venv\Scripts\python.exe -m pip install -r requirements-deep.txt
}
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then run:             python run_all.py"
