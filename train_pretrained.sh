out=egs/main_dataset/
mkdir -p $out
python -m denoiser.audio $noisy > $out/noisy.json
python -m denoiser.audio $clean > $out/clean.json

# ./train.py continue_pretrained=dns64 demucs.hidden=64
