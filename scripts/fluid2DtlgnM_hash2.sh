python main.py fluid \
    --tag fluid2d_tlgnM_hash3 \
    --init_cond taylorgreen_multi \
    --num_hidden_layers 3 \
    --hidden_features 32 \
    -sr 128 \
    -vr 32 \
    --dt 0.05 \
    -T 20 \
    -g 1 \
    --max_n_iter 40000\
    --n_levels 7 \
    --n_features_per_level 2 \
    --log2_hashmap_size 19 \
    --base_resolution 4 \
    --finest_resolution 256 \
    --mlp_unit "[32,32]" \
    --network hashgrid