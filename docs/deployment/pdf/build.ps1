# Build CMW RAG deployment architecture PDF
# Requires: cbap-mkdocs-ru venv (.venv), GTK3 (WeasyPrint on Windows)
# Usage: .\build.ps1
#
# Source: ../deployment_architecture.ru.md (single document, web + print)
# Config: .env in this folder (gitignored) or environment variables
# Output: docs/deployment/pdf/*.pdf (tracked); HTML scratch: <repo>/.scratch/deployment_architecture_pdf/

$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return }
    Get-Content $Path -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith('#')) { return }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { return }
        $key = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim().Trim('"').Trim("'")
        if ($key -and -not [string]::IsNullOrWhiteSpace($value)) {
            Set-Item -Path "Env:$key" -Value $value
        }
    }
}

function Resolve-CbapMkdocsRoot {
    param([System.IO.DirectoryInfo]$CmwRagRoot)

    if ($env:CBAP_MKDOCS_ROOT) {
        $candidate = Resolve-Path -LiteralPath $env:CBAP_MKDOCS_ROOT -ErrorAction SilentlyContinue
        if ($candidate -and (Test-Path (Join-Path $candidate "mkdocs_common.yml"))) {
            return $candidate.Path
        }
        throw "CBAP_MKDOCS_ROOT is set but mkdocs_common.yml was not found: $env:CBAP_MKDOCS_ROOT"
    }

    $reposRoot = $CmwRagRoot.Parent
    # Prefer sibling checkout (active dev); fall back to vendored .reference-repos copy.
    $candidates = @(
        (Join-Path $reposRoot.FullName "cbap-mkdocs-ru"),
        (Join-Path $CmwRagRoot.FullName ".reference-repos\cbap-mkdocs-ru")
    )

    foreach ($path in $candidates) {
        if (Test-Path (Join-Path $path "mkdocs_common.yml")) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }

    throw @"
cbap-mkdocs-ru not found. Set CBAP_MKDOCS_ROOT in docs/deployment/pdf/.env or the environment.
Tried:
  $($candidates -join "`n  ")
Copy docs/deployment/pdf/.env.example to docs/deployment/pdf/.env and set CBAP_MKDOCS_ROOT.
"@
}

$pdfDir = Get-Item (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$deploymentDir = $pdfDir.Parent
$cmwRagRoot = $pdfDir
for ($i = 0; $i -lt 3; $i++) { $cmwRagRoot = $cmwRagRoot.Parent }

Import-DotEnv (Join-Path $pdfDir.FullName ".env")

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$buildScratch = Join-Path $cmwRagRoot.FullName ".scratch\deployment_architecture_pdf"
New-Item -ItemType Directory -Force -Path $buildScratch | Out-Null

$env:MKDOCS_DOCS_DIR = $deploymentDir.FullName
$siteDir = Join-Path $buildScratch ".site"
$env:MKDOCS_SITE_DIR = $siteDir

$sourceMd = Join-Path $deploymentDir.FullName "deployment_architecture.ru.md"
if (-not (Test-Path $sourceMd)) {
    throw "Source document not found: $sourceMd"
}

$cbapRoot = Resolve-CbapMkdocsRoot -CmwRagRoot $cmwRagRoot

$env:MKDOCS_COMMON = Join-Path $cbapRoot "mkdocs_common.yml"
$env:MKDOCS_OVERRIDES = Join-Path $cbapRoot "overrides"
$env:MKDOCS_PDF_TEMPLATES = Join-Path $cbapRoot "pdf_templates"
$env:MKDOCS_SNIPPETS = Join-Path $cbapRoot "docs\ru\.snippets\"
$venvPath = Join-Path $cbapRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPath)) {
    throw "Python venv not found at $venvPath. Create .venv in cbap-mkdocs-ru."
}

$dateStr = Get-Date -Format "yyyy.MM.dd"
$pdfDateIso = if ($env:PDF_DATE) { $env:PDF_DATE } else { Get-Date -Format "yyyy.MM.dd" }
$pdfAuthor = if ($env:PDF_AUTHOR) { $env:PDF_AUTHOR } else { "Составил Артём Седов" }
# Cover metadata (title, subtitle, logo, header) lives in mkdocs.yml; only footer date is injected here.
$env:MKDOCS_PDF_FOOTER_LEFT = "$pdfAuthor $pdfDateIso"
$pdfFileName = "CMW RAG. Архитектура развёртывания. $dateStr.pdf"
$pdfPath = Join-Path $pdfDir.FullName $pdfFileName
# Relative to site_dir (.scratch/.../.site) → docs/deployment/pdf/
$env:MKDOCS_PDF_OUTPUT_FILENAME = "../../../docs/deployment/pdf/$pdfFileName"

$gtkBin = $env:GTK3_BIN
if (-not $gtkBin -and $env:WEASYPRINT_DLL_DIRECTORIES) {
    $gtkBin = $env:WEASYPRINT_DLL_DIRECTORIES
}
if ($gtkBin -and (Test-Path $gtkBin)) {
    $env:WEASYPRINT_DLL_DIRECTORIES = $gtkBin
    if ($env:PATH -notlike "*$gtkBin*") {
        $env:PATH = "$gtkBin;$env:PATH"
    }
}

$configPath = Join-Path $pdfDir.FullName "mkdocs.yml"

Write-Host "Building deployment architecture PDF..." -ForegroundColor Cyan
Write-Host "source: $sourceMd" -ForegroundColor DarkGray
Write-Host "cbap-mkdocs-ru: $cbapRoot" -ForegroundColor DarkGray
Write-Host "build scratch: $buildScratch" -ForegroundColor DarkGray
Push-Location $pdfDir.FullName
try {
    & $venvPath -m mkdocs build -f $configPath --clean
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "PDF output: $pdfPath" -ForegroundColor Green
}
finally {
    Pop-Location
}
