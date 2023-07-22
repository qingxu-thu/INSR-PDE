python main.py flow_fluid \
    --tag flow_fluid_hash2 \
    --num_hidden_layers 3 \
    --hidden_features 32 \
    -sr 128 \
    -vr 32 \
    --dt 0.05 \
    -T 100 \
    -g 0 \
    --nonlinearity sine\
    --lr 1e-4\
    --n_levels 16 \
    --n_features_per_level 8 \
    --log2_hashmap_size 19 \
    --base_resolution 4 \
    --finest_resolution 64 \
    --mlp_unit "[32,32,32]" \
    --optim_type exp \
    --network hashgrid \
    --factor 1 \
    --ent_v 0.8