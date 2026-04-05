$ErrorActionPreference = "Stop"
cd c:\Users\lenovo\OneDrive\Desktop\S\ResolveFlow

# We must use Python 3.12 because Python 3.14 alpha lacks wheels for pydantic and fastapi
if (Test-Path "venv") {
    Write-Host "Removing bad venv..."
    Remove-Item -Recurse -Force venv
}

Write-Host "Creating Python 3.12 virtual environment..."
py -3.12 -m venv venv

Write-Host "Installing dependencies..."
.\venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Starting server on port 7860..."
.\venv\Scripts\python.exe app.py
