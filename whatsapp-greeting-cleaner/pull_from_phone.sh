#!/usr/bin/env bash
# Copia a pasta WhatsApp Images do celular (via USB + depuração USB) para ./whatsapp_images
set -euo pipefail

PHONE_DIR="/sdcard/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images"
LOCAL_DIR="${1:-whatsapp_images}"

adb get-state >/dev/null || { echo "Celular não encontrado. Ative a depuração USB e autorize este PC."; exit 1; }

echo "Arquivos no celular: $(adb shell "find '$PHONE_DIR' -type f | wc -l")"
mkdir -p "$LOCAL_DIR"
# adb pull cria a pasta 'WhatsApp Images' dentro do destino; o conteúdo vai para LOCAL_DIR.
adb pull "$PHONE_DIR/." "$LOCAL_DIR"
echo "Arquivos copiados: $(find "$LOCAL_DIR" -type f | wc -l)"
