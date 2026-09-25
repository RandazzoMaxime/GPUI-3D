param([string]$Exe, [string]$Out, [int]$Wait = 8)
# Lance l'exe, attend, capture le contenu de SA fenêtre (PrintWindow, insensible aux
# fenêtres superposées), relève les logs, tue le process et attend sa fin.
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System; using System.Runtime.InteropServices;
public class W32 {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  public struct RECT { public int L, T, R, B; }
}
"@
[W32]::SetProcessDPIAware() | Out-Null
$log = "$Out.log"
$p = Start-Process -FilePath $Exe -PassThru -RedirectStandardError $log -RedirectStandardOutput "$Out.out"
Start-Sleep -Seconds $Wait
$p.Refresh()
if ($p.HasExited) { "EXITED code=$($p.ExitCode)"; Get-Content $log -Tail 40; exit 1 }
$h = $p.MainWindowHandle
$r = New-Object W32+RECT
[W32]::GetWindowRect($h, [ref]$r) | Out-Null
$bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $g.GetHdc()
[W32]::PrintWindow($h, $hdc, 2) | Out-Null  # PW_RENDERFULLCONTENT
$g.ReleaseHdc($hdc)
$bmp.Save($Out)
"captured $($bmp.Width)x$($bmp.Height) -> $Out"
Stop-Process -Id $p.Id -Force
$p.WaitForExit()
Get-Content $log -Tail 30
