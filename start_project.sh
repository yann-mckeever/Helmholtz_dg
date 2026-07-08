#!/bin/bash

SESSION_NAME="fenicsx-complex"
CONDA_ENV="base" # Remplace par ton environnement si besoin
PROJECT_DIR=~/Documents/PRE/CODE/

source ~/miniforge3/etc/profile.d/conda.sh

# 1. Crée une nouvelle session en arrière-plan et se place dans le dossier
tmux new-session -d -s $SESSION_NAME -c $PROJECT_DIR

tmux rename-window -t $SESSION_NAME:1 "Code"

# 2. Dans le premier panneau (Fenêtre 1, Panneau 1) : On active conda et on lance Neovim
tmux send-keys -t $SESSION_NAME:1 "source ~/miniforge3/etc/profile.d/conda.sh" C-m
tmux send-keys -t $SESSION_NAME:1 "conda activate base && conda activate fenicsx-complex" C-m
tmux send-keys -t $SESSION_NAME:1 "nvim -p main.py src/helmholtz_dg/*.py" C-m

tmux new-window -t $SESSION_NAME -n "Terminal" -c "$PROJECT_DIR"

tmux send-keys -t $SESSION_NAME:2 "source ~/miniforge3/etc/profile.d/conda.sh" C-m
tmux send-keys -t $SESSION_NAME:2 "conda activate base && conda activate fenicsx-complex" C-m
tmux send-keys -t $SESSION_NAME:2 "clear" C-m

tmux attach-session -t $SESSION_NAME
if [ -z "$TMUX" ]; then
  # Si on est sur un terminal classique, on s'attache normalement
  tmux attach-session -t $SESSION_NAME
else
  # Si on est DÉJÀ dans Tmux, on bascule vers la nouvelle session sans planter
  tmux switch-client -t $SESSION_NAME
fi
