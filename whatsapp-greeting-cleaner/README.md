# WhatsApp Greeting Cleaner

Encontra e apaga as imagens de "bom dia", "boa noite", anjinhos, desenhos de flores e fotos
de bebê de banco de imagem que lotam a pasta `WhatsApp Images` (6 GB, 38 mil arquivos).

Tem duas soluções: uma só no celular e uma no PC com Linux, usando classificação de imagens
zero-shot com CLIP.

---

## Passo 0: parar de chegar mais (faça primeiro, no celular)

O WhatsApp **não tem download automático por grupo**. A configuração é global. Por grupo,
dá para controlar só se a mídia aparece na galeria. Então:

1. **Achar os grupos:** na busca do WhatsApp, digite `HOG` e anote os grupos. Faça o mesmo com `CIA Alves`.
2. **Em cada um desses grupos:** toque no nome do grupo → **Visibilidade de mídia** → **Não**.
   As imagens desses grupos deixam de ir para a galeria e para o backup do Google Fotos.
3. **Opcional (mais radical):** Configurações → Armazenamento e dados → Download automático
   de mídia → desmarque **Fotos** em Dados móveis, Wi-Fi e Roaming. Vale para todos os
   contatos: as fotos só baixam quando você tocar nelas.

> Atenção com os grupos HOG: fotos de rolê e de encontros também ficam escondidas da galeria.
> Para salvar uma específica, abra a foto → ⋮ → Salvar.

---

## Solução 1: só no celular (cerca de 30 min, sem código)

- [ ] **WhatsApp → Configurações → Armazenamento e dados → Gerenciar armazenamento**
  - Abra **"Encaminhadas com frequência"**. É onde as correntes de bom dia ficam. Revise, selecione tudo e apague.
  - Na lista de conversas embaixo, abra **CIA Alves** → selecione tudo → apague.
    Nos grupos HOG, selecione só o que for corrente, porque é ali que estão as fotos de rolê.
- [ ] **Google Fotos** (ele lê texto e reconhece objetos nas fotos):
  - Pesquise `bom dia`, depois `boa noite`, `anjo`, `flores` e `bebê`.
  - Toque e segure na primeira imagem e arraste para selecionar várias → **Excluir**.
  - Excluir pelo app do Google Fotos apaga **do celular e da nuvem ao mesmo tempo**.
  - Depois esvazie a **Lixeira** (senão o espaço só volta em 60 dias).
- [ ] **Backup do WhatsApp:** Configurações → Conversas → Backup de conversas → **Fazer backup**.
  O Google Drive guarda só o último backup, então o novo já sai sem as imagens apagadas.

**Limitação:** a busca do Google Fotos não distingue um desenho de flor de uma foto de flor, nem
o seu bebê de um bebê de banco de imagem. Revise antes de excluir.

---

## Solução 2: no PC com Linux (projeto de ML)

### Como funciona

```
celular ──adb pull──▶ whatsapp_images/ ──scan (CLIP)──▶ results.csv
                                                            │
             review.html ◀── quarantine/<categoria>/ ◀──quarantine (limiares)
                                   │
                    você revisa e tira o que é para manter
                                   │
celular ◀──adb rm── delete_on_phone.sh (apaga o que sobrou na quarentena)
```

- **CLIP zero-shot:** o modelo compara cada imagem com frases ("a 'good morning' greeting card",
  "a drawing of flowers", "a family photo"...) e diz qual frase combina mais. Não precisa
  treinar nada nem rotular imagens.
- **Categorias-alvo:** `saudacao`, `anjo`, `flores_desenho`, `bebe`.
- **Categorias de manter:** `pessoal`, `flores_foto`, `moto`, `documento`, `outros`. Elas existem
  para que fotos reais tenham para onde ir. Sem elas, tudo vira "bom dia".
- **Desenho × foto de flor:** são categorias separadas, e o CLIP distingue bem estilo
  (ilustração × fotografia).
- **Bebê:** usa um limiar mais alto (0.75) e fica numa pasta própria, porque é o maior risco
  de apagar foto da família. Revise essa pasta com calma.
- **OCR (opcional, `--ocr`):** o Tesseract lê o texto da imagem. Se aparecer "bom dia", "boa noite",
  "abençoado" etc., a imagem vai para `saudacao`. O OCR só roda nas imagens que o CLIP já achou
  suspeitas, porque rodar em todas levaria horas.
- **Duas etapas:** o `scan` grava as probabilidades no CSV. O `quarantine` aplica os limiares.
  Dá para testar outros limiares sem rodar o modelo de novo.

### Passo a passo

```bash
# 1. Preparar (uma vez)
sudo apt install adb tesseract-ocr tesseract-ocr-por
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Celular: Configurações → Sobre o telefone → Informações de software →
#    toque 7x em "Número de compilação". Depois Opções do desenvolvedor → Depuração USB.
#    Conecte no USB e autorize o PC.
./pull_from_phone.sh                 # copia ~6 GB para whatsapp_images/

# 3. Classificar (CPU: ~30–60 min para 38 mil imagens; se cair, rode de novo que ele retoma)
python classify.py scan whatsapp_images/ --out results.csv --ocr

# 4. Separar para a quarentena
python classify.py quarantine whatsapp_images/ --csv results.csv --to quarantine/
#    mais rígido:  --threshold 0.6     só bebê mais rígido:  --set bebe=0.9

# 5. Revisar: abra quarantine/review.html (as mais duvidosas aparecem primeiro).
#    O que você quiser MANTER, tire da pasta quarantine/.

# 6. Apagar do celular o que sobrou na quarentena (pede confirmação digitando APAGAR)
./delete_on_phone.sh quarantine/
```

Para errar menos, use o modelo maior (mais lento):
`--model openai/clip-vit-large-patch14`.

### E o Google Fotos / backup?

- O `delete_on_phone.sh` apaga **só do celular**. Se a pasta WhatsApp Images estava com backup
  no Google Fotos (Fotos → foto de perfil → Backup → Fazer backup de pastas do dispositivo), as
  cópias na nuvem continuam lá.
- A API do Google Fotos **não permite apagar fotos**, por decisão do Google. Para a nuvem, use a
  busca no [photos.google.com](https://photos.google.com) como na Solução 1.
- O backup do WhatsApp se resolve fazendo um backup novo depois da limpeza.

### Privacidade

Tudo roda localmente. Nenhuma imagem sai do seu PC. O `.gitignore` impede que fotos, a quarentena
e os CSVs entrem no git.

---

## Para o portfólio / entrevista

Uma forma de contar (formato STAR):

- **Situação:** 6 GB e 38 mil imagens de WhatsApp, a maioria correntes de "bom dia" misturadas
  com fotos pessoais.
- **Tarefa:** separar o lixo sem apagar fotos da família e sem rotular dados para treinar.
- **Ação:** classificação zero-shot com CLIP, com categorias de contraste para reduzir falso
  positivo. OCR (Tesseract) como segundo sinal, só nos casos suspeitos. Pipeline em duas etapas
  (inferência → CSV → regra de decisão), para ajustar limiares sem reprocessar. Processamento
  retomável. Revisão humana antes da ação irreversível.
- **Resultado:** N GB liberados (preencha com o seu número). Dá para medir a precisão rotulando
  200 imagens aleatórias do `results.csv`.

Próximos passos possíveis: remover duplicatas com perceptual hash (as correntes se repetem muito
entre grupos), um classificador linear em cima dos embeddings do CLIP usando as suas correções da
revisão (few-shot) e rodar com GPU.
