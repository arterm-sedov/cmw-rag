# Build executive report PDF
# Requires: cbap-mkdocs-ru venv (Python 3.13), GTK3, Node.js + mmdc
# Usage: .\build_executive_report_pdf.ps1
#
# Paths resolved relative to this script location

$scriptDir = Get-Item (Split-Path -Parent $MyInvocation.MyCommand.Definition)

# Navigate from script folder up to 4 levels up
# executive-research-technology-transfer -> research -> docs -> cmw-rag -> Repos
$reposRoot = $scriptDir
for ($i = 0; $i -lt 4; $i++) {
    $reposRoot = $reposRoot.Parent
}

$cbapRoot = Join-Path $reposRoot.FullName "cbap-mkdocs-ru"

# Set env vars for YAML !ENV tags (fallback to relative paths if not set)
$env:MKDOCS_COMMON = Join-Path $cbapRoot "mkdocs_common.yml"
$env:MKDOCS_OVERRIDES = Join-Path $cbapRoot "overrides"
$env:MKDOCS_PDF_TEMPLATES = Join-Path $cbapRoot "pdf_templates"
$env:MKDOCS_SNIPPETS = Join-Path $cbapRoot "docs\ru\.snippets/"
$venvPath = Join-Path $cbapRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $scriptDir.FullName "mkdocs_executive_report_pdf.yml"
$outputPath = Join-Path $scriptDir.FullName ".site\Comindware. Коммерческое обоснование внедрения ИИ.pdf"

Write-Host "Building executive report PDF..." -ForegroundColor Cyan
& $venvPath -m mkdocs build -f $configPath
Write-Host "PDF output: $outputPath" -ForegroundColor Green
