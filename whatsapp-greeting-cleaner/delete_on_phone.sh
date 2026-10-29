#!/usr/bin/env bash
# Apaga do celular tudo o que continua na quarentena (depois da sua revisão).
# A estrutura é quarantine/<categoria>/<caminho dentro de WhatsApp Images>.
set -euo pipefail

PHONE_DIR="/sdcard/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images"
QUARANTINE="${1:-quarantine}"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

adb get-state >/dev/null || { echo "Celular não encontrado. Ative a depuração USB e autorize este PC."; exit 1; }

(cd "$QUARANTINE" && find . -mindepth 2 -type f ! -name review.html | cut -d/ -f3-) | sort -u > "$LIST"
count=$(wc -l < "$LIST")
[ "$count" -gt 0 ] || { echo "Quarentena vazia, nada a apagar."; exit 0; }

echo "Exemplos:"; head -5 "$LIST"
read -r -p "Apagar $count arquivos do celular? Isso não tem volta. Digite APAGAR: " answer
[ "$answer" = "APAGAR" ] || { echo "Cancelado."; exit 1; }

adb push "$LIST" /data/local/tmp/greetings.txt >/dev/null
adb shell "cd '$PHONE_DIR' && while IFS= read -r f; do rm -f -- \"\$f\"; done < /data/local/tmp/greetings.txt; rm /data/local/tmp/greetings.txt"

# Avisa a galeria/Google Fotos que os arquivos sumiram (Android 11+). Se falhar, reinicie o celular.
adb shell content call --method scan_volume --uri content://media --arg external_primary >/dev/null 2>&1 || true

echo "Feito. Restam no celular: $(adb shell "find '$PHONE_DIR' -type f | wc -l") arquivos."
