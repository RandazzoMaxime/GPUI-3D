param([string]$A, [string]$B, [int]$SkipTop = 0, [string]$Map = "")
# Per-pixel difference over the whole image (max of R,G,B). Different sizes = failure.
# -Map: writes the difference amplified x10 as grayscale, and the box of pixels > 2 LSB.
Add-Type -AssemblyName System.Drawing
function Pixels($path) {
    $bmp = [System.Drawing.Bitmap]::FromFile($path)
    $rect = New-Object System.Drawing.Rectangle 0, 0, $bmp.Width, $bmp.Height
    $data = $bmp.LockBits($rect, 'ReadOnly', 'Format32bppArgb')
    $bytes = New-Object byte[] ($data.Stride * $bmp.Height)
    [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $bytes, 0, $bytes.Length)
    $r = @{ bytes = $bytes; w = $bmp.Width; h = $bmp.Height }
    $bmp.UnlockBits($data); $bmp.Dispose(); $r
}
$pa = Pixels $A; $pb = Pixels $B
if ($pa.w -ne $pb.w -or $pa.h -ne $pb.h) { "{0,-24} TAILLES {1}x{2} vs {3}x{4}" -f (Split-Path $B -Leaf), $pa.w, $pa.h, $pb.w, $pb.h; return }
$max = 0; $gt2 = 0
$minX = $pa.w; $minY = $pa.h; $maxX = -1; $maxY = -1
$out = if ($Map) { New-Object byte[] $pa.bytes.Length } else { $null }
for ($i = $SkipTop * $pa.w * 4; $i -lt $pa.bytes.Length; $i += 4) {
    $d = 0
    for ($c = 0; $c -lt 3; $c++) { $v = [Math]::Abs([int]$pa.bytes[$i + $c] - [int]$pb.bytes[$i + $c]); if ($v -gt $d) { $d = $v } }
    if ($d -gt $max) { $max = $d }
    if ($d -gt 2) {
        $gt2++
        $p = $i / 4; $x = $p % $pa.w; $y = [Math]::Floor($p / $pa.w)
        if ($x -lt $minX) { $minX = $x }; if ($x -gt $maxX) { $maxX = $x }
        if ($y -lt $minY) { $minY = $y }; if ($y -gt $maxY) { $maxY = $y }
    }
    if ($out) { $g = [Math]::Min(255, $d * 10); $out[$i] = $g; $out[$i + 1] = $g; $out[$i + 2] = $g; $out[$i + 3] = 255 }
}
"{0,-24} max {1,3} | >2 LSB {2,7} / {3}" -f (Split-Path $B -Leaf), $max, $gt2, ($pa.w * $pa.h)
if ($Map) {
    if ($gt2 -gt 0) { "  box > 2 LSB: x {0}..{1}, y {2}..{3}" -f $minX, $maxX, $minY, $maxY }
    $bmp = New-Object System.Drawing.Bitmap $pa.w, $pa.h, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $rect = New-Object System.Drawing.Rectangle 0, 0, $pa.w, $pa.h
    $data = $bmp.LockBits($rect, 'WriteOnly', 'Format32bppArgb')
    [System.Runtime.InteropServices.Marshal]::Copy($out, 0, $data.Scan0, $out.Length)
    $bmp.UnlockBits($data); $bmp.Save($Map); $bmp.Dispose()
}
