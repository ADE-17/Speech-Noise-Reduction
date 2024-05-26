oisy_train=path to valentini
clean_train=path to valentini
noisy_test=path to valentini
clean_test=path to valentini
noisy_dev=path to valentini
clean_dev=path to valentini

mkdir -p egs/val/tr
mkdir -p egs/val/cv
mkdir -p egs/val/tt

python -m denoiser.audio $noisy_train > egs/val/tr/noisy.json
python -m denoiser.audio $clean_train > egs/val/tr/clean.json

python -m denoiser.audio $noisy_test > egs/val/tt/noisy.json
python -m denoiser.audio $clean_test > egs/val/tt/clean.json

python -m denoiser.audio $noisy_dev > egs/val/cv/noisy.json
python -m denoiser.audio $clean_dev > egs/val/cv/clean.json