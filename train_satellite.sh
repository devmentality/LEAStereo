  if [ "$#" -ne 1 ]; then
    echo "Provide experiment name"
fi


CUDA_VISIBLE_DEVICES="0,1,2" python3 train.py --batch_size=3 \
                --crop_height=384 \
                --crop_width=576 \
                --maxdisp=192 \
                --threads=8 \
                --save_path='./run/' \
                --fea_num_layer 6 --mat_num_layers 12 \
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='run/sceneflow/best/architecture/feature_network_path.npy' \
                --cell_arch_fea='run/sceneflow/best/architecture/feature_genotype.npy' \
                --net_arch_mat='run/sceneflow/best/architecture/matching_network_path.npy' \
                --cell_arch_mat='run/sceneflow/best/architecture/matching_genotype.npy' \
                --dataset=satellite \
                --resume='./run/sceneflow/best/checkpoint/best.pth' \
                --experiment=$1
