gpu=0
echo $gpu
# CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=Biological -phase=1
# CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=Biological -phase=2
# CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=Biological -phase=3
# CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=Drug -phase=1
# CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=Drug -phase=2
# CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=Drug -phase=3
# CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=all -phase=1
# CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=all -phase=2
# CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=all -phase=3
CUDA_VISIBLE_DEVICES=$gpu python model.py -train_mode=all -phase=all
