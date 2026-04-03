#!/bin/bash
# Gera exemplos de todas as transformacoes do Image CLIpper
# Uso: cd proc_imagens && source venv/bin/activate && bash tb1/run_examples.sh

DIR="tb1/imagens"
OUT="tb1/imagens/resultados"
mkdir -p "$OUT"

IMG1="$DIR/WhatsApp Image 2026-04-03 at 18.45.15.jpeg"
IMG2="$DIR/WhatsApp Image 2026-04-03 at 18.45.15 (1).jpeg"
IMG3="$DIR/WhatsApp Image 2026-04-03 at 18.45.15 (2).jpeg"
IMG4="$DIR/WhatsApp Image 2026-04-03 at 18.45.15 (3).jpeg"
IMG5="$DIR/WhatsApp Image 2026-04-03 at 18.45.15 (4).jpeg"
IMG6="$DIR/WhatsApp Image 2026-04-03 at 18.45.16.jpeg"
IMG7="$DIR/WhatsApp Image 2026-04-03 at 18.45.16 (1).jpeg"

# === TRANSFORMACOES DE INTENSIDADE ===

python tb1/editor.py inverse  -i "$IMG1" -o "$OUT/img1_inverse.jpeg"
python tb1/editor.py log      -i "$IMG2" -o "$OUT/img2_log.jpeg"
python tb1/editor.py gamma    -i "$IMG3" -o "$OUT/img3_gamma_35.jpeg"  --gamma 3.5
python tb1/editor.py gamma    -i "$IMG4" -o "$OUT/img4_gamma_22.jpeg"  --gamma 2.2
python tb1/editor.py contrast -i "$IMG5" -o "$OUT/img5_contrast.jpeg"  --intervals "0,50,200,255" --targets "0,150,200,255"
python tb1/editor.py creative -i "$IMG6" -o "$OUT/img6_solarize.jpeg"  --threshold 128

# === TRANSFORMACOES GEOMETRICAS ===

python tb1/editor.py translate -i "$IMG7" -o "$OUT/img7_translate.jpeg"        --dx 200 --dy 150
python tb1/editor.py rotate    -i "$IMG1" -o "$OUT/img1_rotate_30.jpeg"        --angle 30 --force
python tb1/editor.py rotate    -i "$IMG2" -o "$OUT/img2_rotate_autozoom.jpeg"  --angle 30 --strategy autozoom
python tb1/editor.py rotate    -i "$IMG3" -o "$OUT/img3_rotate_nearest.jpeg"   --angle 45 --strategy nearest
python tb1/editor.py scale     -i "$IMG4" -o "$OUT/img4_scale_4x.jpeg"         --factor 4.0
python tb1/editor.py scale     -i "$IMG5" -o "$OUT/img5_scale_010.jpeg"        --factor 0.10
python tb1/editor.py crop      -i "$IMG6" -o "$OUT/img6_crop.jpeg"             --x 200 --y 600 --w 500 --h 400

echo "Resultados salvos em $OUT/"
