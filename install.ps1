# Script de instalación para ECG Backend
# Requiere Python 3.12+

Write-Host "Creando entorno virtual..." -ForegroundColor Cyan
python -m venv venv

Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

Write-Host "Actualizando pip, setuptools y wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host "Instalando PyTorch (CPU)..." -ForegroundColor Cyan
pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "Instalando dependencias..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n✅ Instalación completa. Ejecuta: python main.py" -ForegroundColor Green
