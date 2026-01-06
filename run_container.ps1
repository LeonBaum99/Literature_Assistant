param (
    [Switch]$Rebuild  # Use: .\run_container.ps1 -Rebuild
)

$APP_NAME = "paper-rag-app"

# --- 1. Detect Hardware ---
if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
    $TAG = "gpu"
    $BUILD_ARG = "gpu"
    Write-Host "✅ NVIDIA GPU Detected (Mode: GPU)" -ForegroundColor Green
} else {
    $TAG = "cpu"
    $BUILD_ARG = "cpu"
    Write-Host "⚠️  No NVIDIA GPU found (Mode: CPU)" -ForegroundColor Yellow
}

$IMAGE_FULL_NAME = "$($APP_NAME):$($TAG)"

# --- 2. Build Strategy ---
$ImageExists = docker images -q $IMAGE_FULL_NAME
$ShouldBuild = (-not $ImageExists) -or $Rebuild

if ($ShouldBuild) {
    Write-Host "🛠️  Building Image ($IMAGE_FULL_NAME)..." -ForegroundColor Cyan
    docker build -t $IMAGE_FULL_NAME -f Dockerfile --build-arg DEVICE_TYPE=$BUILD_ARG .
    if ($LASTEXITCODE -ne 0) { Write-Host "❌ Build Failed!" -ForegroundColor Red; exit }
} else {
    Write-Host "⏩ Image found! Skipping build. (Use -Rebuild to force update)" -ForegroundColor Gray
}

# --- 3. Run Container (Dev Mode) ---
Write-Host "🚀 Starting Container with Hot Reload..." -ForegroundColor Green

$DockerArgs = @(
    "run", "--rm", "-it",
    "-p", "8000:8000",
    "-v", "${PWD}/chroma_db:/app/chroma_db",
    "-v", "${PWD}:/app"
)

# --- NEW: Inject .env file if it exists ---
if (Test-Path ".env") {
    Write-Host "📄 Found .env file, injecting environment variables..." -ForegroundColor Cyan
    $DockerArgs += "--env-file"
    $DockerArgs += ".env"
}

if ($TAG -eq "gpu") {
    $DockerArgs += "--gpus"
    $DockerArgs += "all"
}

$DockerArgs += $IMAGE_FULL_NAME

$DockerArgs += "uvicorn"
$DockerArgs += "backend.main:app"
$DockerArgs += "--host"
$DockerArgs += "0.0.0.0"
$DockerArgs += "--port"
$DockerArgs += "8000"
$DockerArgs += "--reload"

& docker @DockerArgs