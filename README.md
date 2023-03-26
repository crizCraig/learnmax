Setup

1. conda create --name lm python=3.10
1. Install Pytorch with conda via official instructions, e.g.
   ```
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```
1. `conda install scipy`
1. `pip install -r requirements.txt`
1. [Install Atari-Py roms](https://github.com/openai/atari-py#roms) or maybe use just `pip install -U gym[atari]`
1. `pip install -U gym[atari]` if you didn't on the previous step. Also perhaps `ale-import-roms --import-from-pkg atari_py.atari_roms` or just `ale-import-roms`

Checkpoints

- [DVQ Montezuma's revenge](https://drive.google.com/file/d/1pKsos7N3CrnnCt92uoz_poIQAl0ilNVB/view?usp=sharing)
