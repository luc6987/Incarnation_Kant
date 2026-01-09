#!/bin/bash
export STREAMLIT_CONFIG_DIR=/Data/Incarnation_Kant/.streamlit
export HF_HOME=/Data/Incarnation_Kant/.venv/.cache/huggingface
export TORCH_HOME=/Data/Incarnation_Kant/.venv/.cache/torch
export XDG_CACHE_HOME=/Data/Incarnation_Kant/.venv/.cache
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p $XDG_CACHE_HOME

./.venv/bin/streamlit run app/main.py
