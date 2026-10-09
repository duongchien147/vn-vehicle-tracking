# Biên dịch báo cáo LaTeX (XeLaTeX) ra PDF trên máy, vì Overleaf miễn phí giới hạn 10 giây mỗi lần biên dịch
# không đủ cho cả báo cáo. Soạn trên Overleaf, tải mã nguồn về (Menu -> Download -> Source), rồi chạy:
#
#   .\report\build_pdf.ps1 -Zip "$env:USERPROFILE\Downloads\bao_cao_overleaf.zip"   # lấy bản mới từ Overleaf
#   .\report\build_pdf.ps1                                                       # biên dịch report\overleaf hiện có
#
# Kết quả: report\bao_cao.pdf. Lần đầu tự tải Tectonic (trình biên dịch XeTeX, ~20 MB) về tools\tectonic\;
# Tectonic tự tải các gói LaTeX cần dùng ở lần chạy đầu (vài phút), các lần sau ~10 giây.
param(
    [string]$Zip = ""
)
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path $PSScriptRoot "overleaf"
$BuildDir = Join-Path $PSScriptRoot "build"
$OutputPdf = Join-Path $PSScriptRoot "bao_cao.pdf"
$TectonicDir = Join-Path $ProjectRoot "tools\tectonic"
$Tectonic = Join-Path $TectonicDir "tectonic.exe"
$TectonicVersion = "0.17.0"
$TectonicUrl = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40$TectonicVersion/tectonic-$TectonicVersion-x86_64-pc-windows-msvc.zip"

function Install-Tectonic {
    if (Test-Path $Tectonic) { return }
    Write-Host "Tải Tectonic $TectonicVersion về $TectonicDir ..."
    New-Item -ItemType Directory -Force $TectonicDir | Out-Null
    $archive = Join-Path $TectonicDir "tectonic.zip"
    Invoke-WebRequest -Uri $TectonicUrl -OutFile $archive -UseBasicParsing
    Expand-Archive -Path $archive -DestinationPath $TectonicDir -Force
    Remove-Item $archive
}

function Update-SourceFromZip([string]$zipPath) {
    if (-not (Test-Path $zipPath)) { throw "Không thấy file zip: $zipPath" }
    # Giải nén ra thư mục tạm trước, chỉ thay report\overleaf khi giải nén thành công; giữ bản cũ ở overleaf_truoc
    $staging = Join-Path $PSScriptRoot "overleaf_moi"
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
    Expand-Archive -Path $zipPath -DestinationPath $staging
    if (-not (Test-Path (Join-Path $staging "main.tex"))) { throw "File zip không có main.tex ở thư mục gốc" }
    $previous = Join-Path $PSScriptRoot "overleaf_truoc"
    if (Test-Path $previous) { Remove-Item -Recurse -Force $previous }
    if (Test-Path $SourceDir) { Rename-Item $SourceDir "overleaf_truoc" }
    Rename-Item $staging "overleaf"
    Write-Host "Đã cập nhật report\overleaf từ $zipPath (bản trước giữ ở report\overleaf_truoc)"
}

Install-Tectonic
if ($Zip) { Update-SourceFromZip $Zip }

$main = Join-Path $SourceDir "main.tex"
if (Select-String -Path $main -Pattern '^\s*\\includeonly' -Quiet) {
    Write-Warning "main.tex đang có dòng \includeonly chưa bị chú thích: PDF sẽ chỉ có phần đó. Xoá dòng này để ra cả báo cáo."
}

New-Item -ItemType Directory -Force $BuildDir | Out-Null
Write-Host "Biên dịch $main ..."
Push-Location $SourceDir
try {
    # Tectonic tự chạy BibTeX và biên dịch lại đủ lượt; thông báo lỗi BibTeX trên file .aux của từng chương
    # (\include) và cảnh báo Fontconfig là bình thường. Tectonic in tiến trình ra stderr, mà PowerShell 5.1 coi
    # stderr là lỗi khi ErrorActionPreference = Stop, nên tạm chuyển sang Continue và chỉ xét mã thoát.
    $ErrorActionPreference = "Continue"
    & $Tectonic -X compile --keep-logs --outdir $BuildDir main.tex 2>&1 | ForEach-Object { "$_" }
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($exitCode -ne 0) { throw "Biên dịch lỗi, xem $BuildDir\main.log" }
} finally {
    Pop-Location
}
Copy-Item (Join-Path $BuildDir "main.pdf") $OutputPdf -Force
$undefined = (Select-String -Path (Join-Path $BuildDir "main.log") -Pattern "undefined" | Measure-Object).Count
Write-Host "Xong: $OutputPdf (cảnh báo tham chiếu/trích dẫn chưa xác định: $undefined)"
