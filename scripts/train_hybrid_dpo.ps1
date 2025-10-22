Param(
  [string]$Python = "python"
)

Write-Host "[train_hybrid_dpo.ps1] Starting training via $Python"

$code = @'
from src.hybrid_dpo import train_hybrid_dpo

train_hybrid_dpo({})
'@

& $Python -c $code

Write-Host "[train_hybrid_dpo.ps1] Done"


