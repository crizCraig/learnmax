Setup

1. Create conda env with ~~requirements.yml~~ Pytorch official instructions
2. `conda install scipy`
3. `pip install -r requirements.txt`
4. [Install Atari-Py roms](https://github.com/openai/atari-py#roms) or maybe use just `pip install -U gym[atari]`
5. `pip install -U gym[atari]` if you didn't on the previous step. Also perhaps `ale-import-roms --import-from-pkg atari_py.atari_roms` or just `ale-import-roms`

Checkpoints

- [DVQ Montezuma's revenge](https://drive.google.com/file/d/1pKsos7N3CrnnCt92uoz_poIQAl0ilNVB/view?usp=sharing)
