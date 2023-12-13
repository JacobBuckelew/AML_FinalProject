python -u train_CANF.py\
    --seed=7\
    --gpu=0\
    --model=CANF_rds\
    --dataset=rds\
    --window_size=3\
    --stride_size=3\
    --num_blocks=4\
    --st_units=32\
    --epochs=50\
    --lr=1e-2\
    --entities=10\