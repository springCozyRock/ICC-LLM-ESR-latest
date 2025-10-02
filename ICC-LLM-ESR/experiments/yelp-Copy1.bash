## LLM-ESR -- SASRec, Bert4Rec, GRU4Rec
gpu_id=0
dataset="yelp"
seed_list=(42 43 44)

model_name="llmesr_bert4rec"
mask_prob=0.6
for seed in ${seed_list[@]}
do
        python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 64 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --mask_prob ${mask_prob} \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --ts_user 12 \
                --ts_item 13 \
                --freeze \
                --log \
                --user_sim_func kd \
                --alpha 0.1 \
                --gamma 0.05\
                --use_cross_att
done