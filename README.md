# FGL
The Frequency Global and Local (FGL) context block

# How to run
export CUDA_VISIBLE_DEVICES=0,1

export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=2 --master_port=501357 --attention_type='scsp' --use_env main.py
