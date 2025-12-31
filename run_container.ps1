param (
    [Switch]$Rebuild  # Allows you to force a rebuild using: .\run_container.ps1 -Rebuild
)

$APP_NAME = "paper-rag-app"

# 1. Detect GPU and determine Tag
if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
    $TAG = "gpu"
    $BUILD_ARG = "gpu"
    $GPU_FLAG = "--gpus all"
    Write-Host "✅ NVIDIA GPU Detected (Mode: GPU)" -ForegroundColor Green
} else {
    $TAG = "cpu"
    $BUILD_ARG = "cpu"
    $GPU_FLAG = ""
    Write-Host "⚠️  No NVIDIA GPU found (Mode: CPU)" -ForegroundColor Yellow
}

$IMAGE_FULL_NAME = "$($APP_NAME):$($TAG)"

# 2. Check if image exists
$ImageExists = docker images -q $IMAGE_FULL_NAME
$ShouldBuild = (-not $ImageExists) -or $Rebuild

# 3. Build only if necessary
if ($ShouldBuild) {
    Write-Host "🛠️  Building Image ($IMAGE_FULL_NAME)..." -ForegroundColor Cyan
    
    # We use -f Dockerfile to be safe, assuming you renamed it back to standard
    docker build -t $IMAGE_FULL_NAME -f Dockerfile --build-arg DEVICE_TYPE=$BUILD_ARG .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Build Failed!" -ForegroundColor Red
        exit
    }
} else {
    Write-Host "⏩ Image found! Skipping build. (Use -Rebuild to force update)" -ForegroundColor Gray
}

# 4. Run the Container
Write-Host "🚀 Starting Container..." -ForegroundColor Green

# We allow $GPU_FLAG to be empty for CPU mode
# Invoke-Expression is used here to handle the dynamic GPU flag cleanly
$RunCmd = "docker run --rm -it -p 8000:8000 -v `"${PWD}/chroma_db:/app/chroma_db`"
if ($TAG -eq "gpu") {
    $RunCmd += " --gpus all"
}
$RunCmd += " $IMAGE_FULL_NAME"

Invoke-Expression $RunCmd