param(
    [int]$Port = 8080
)

# Ensure wasm-pack is installed
if (-not (Get-Command wasm-pack -ErrorAction SilentlyContinue)) {
    Write-Host "wasm-pack not found. Installing..." -ForegroundColor Yellow
    cargo install wasm-pack
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install wasm-pack!" -ForegroundColor Red
        exit 1
    }
}

# Ensure wasm32 target is installed
$targets = rustup target list --installed
if ($targets -notcontains "wasm32-unknown-unknown") {
    Write-Host "Adding wasm32-unknown-unknown target..." -ForegroundColor Yellow
    rustup target add wasm32-unknown-unknown
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to add wasm32 target!" -ForegroundColor Red
        exit 1
    }
}

# Build the WASM example
Write-Host "Building WGPUI WASM hello world example..." -ForegroundColor Green

wasm-pack build --target web --out-dir pkg
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Open http://localhost:$Port in a browser that supports WebGPU."
Write-Host ""
Write-Host "Starting HTTP server on port $Port..." -ForegroundColor Yellow

# Try to find a suitable HTTP server
if (Get-Command npx -ErrorAction SilentlyContinue) {
    Write-Host "Using npx serve" -ForegroundColor Cyan
    npx serve . -l $Port --no-clipboard
    return
}

if (Get-Command dotnet -ErrorAction SilentlyContinue) {
    Write-Host "Using dotnet serve" -ForegroundColor Cyan
    dotnet tool install -g dotnet-serve 2>$null
    dotnet serve -p $Port -d .
    return
}

if (Get-Command php -ErrorAction SilentlyContinue) {
    Write-Host "Using PHP built-in server" -ForegroundColor Cyan
    php -S localhost:$Port
    return
}

if (Get-Command node -ErrorAction SilentlyContinue) {
    Write-Host "Using Node.js http-server" -ForegroundColor Cyan
    npx http-server . -p $Port -c-1
    return
}

# Fallback: PowerShell 5+ HTTP listener (basic)
Write-Host "Using PowerShell fallback HTTP server on http://localhost:$Port" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow

$root = (Resolve-Path .).Path
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:$Port/")
$listener.Start()

while ($listener.IsListening) {
    $context = $listener.GetContext()
    $request = $context.Request
    $response = $context.Response

    $path = $request.Url.AbsolutePath.TrimStart('/')
    if ([string]::IsNullOrEmpty($path)) { $path = "index.html" }

    $filePath = Join-Path $root $path
    if (Test-Path $filePath -PathType Leaf) {
        $content = [IO.File]::ReadAllBytes($filePath)
        $contentType = switch ([IO.Path]::GetExtension($path)) {
            '.html' { 'text/html' }
            '.js'   { 'application/javascript' }
            '.wasm' { 'application/wasm' }
            '.css'  { 'text/css' }
            '.png'  { 'image/png' }
            default { 'application/octet-stream' }
        }
        $response.ContentType = $contentType
        $response.ContentLength64 = $content.Length
        $response.OutputStream.Write($content, 0, $content.Length)
    } else {
        $response.StatusCode = 404
        $errorMsg = [Text.Encoding]::UTF8.GetBytes("404 Not Found")
        $response.OutputStream.Write($errorMsg, 0, $errorMsg.Length)
    }
    $response.Close()
}

$listener.Stop()
