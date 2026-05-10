# Solving Sokoban with Painter-Thinker architecture

## Get data

Data available on google drive https://drive.google.com/drive/folders/1Ac9MwbjnS9nFCEOoxBXNGkDYlXoJL_Xe. Download it and extract to `sokoban/data/raw` directory

## Experiments

V0TOK
Thinker wytrenowany osobno na tokenach. Painter v0tok trenowany osobno z pre-trained trm, concat

V1
Painter i Thinker trenowane osobno (painter MSE, thinker tokeny). Maska dla paintera to decoded(solution_representation).

V2
Painter i Thinker trenowane razem. Aby Painter słuchał wciąż niewytrenowanego thinkera - 2 stages of training. Dla każdego batcha najpierw zamrażamy Paintera i aktualizujemy samego Thinkera. Następnie odmrażamy Paintera i aktualizujemy wszystko (tyle że Thinker dostaje na wejściu już rozwiązaną planszę). Trzeba tylko wymyślić żeby dodać karę dla Thinkera żeby nie zaczął “malować” w tym pierwszym stage.

## Train

```bash
wandb login
```

```bash
python3 train_pt.py experiment=sokoban_pt model=pt_v0tok
```

```bash
python3 train_pt.py experiment=sokoban_pt model=pt_v1
```

```bash
python3 train_pt.py experiment=sokoban_pt model=pt_v2
```
