## 仅验证模糊约束实验（修正语法错误）
gpu_id=0
dataset="fashion"
seed_list=(42 43 44)
ts_user=3
ts_item=4

# ================================= 1. SASRec 模型（模糊约束） =================================
model_name="llmesr_sasrec"
for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
        --model_name ${model_name} \
        --hidden_size 64 \
        --train_batch_size 128 \
        --max_len 200 \
        --gpu_id ${gpu_id} \
        --num_workers 8 \
        --num_train_epochs 200 \
        --seed ${seed} \
        --check_path "fuzzy" \
        --patience 20 \
        --ts_user ${ts_user} \
        --ts_item ${ts_item} \
        --freeze \
        --log \
        --user_sim_func kd \
        --alpha 0.1 \
        --gamma 0.01 \
        --use_cross_att \
        --use_fuzzy
done

# ================================= 3. GRU4Rec 模型（模糊约束，可选） =================================
model_name="llmesr_gru4rec"
for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
        --model_name ${model_name} \
        --hidden_size 64 \
        --train_batch_size 128 \
        --max_len 200 \
        --gpu_id ${gpu_id} \
        --num_workers 8 \
        --num_train_epochs 200 \
        --seed ${seed} \
        --check_path "fuzzy" \
        --patience 20 \
        --ts_user ${ts_user} \
        --ts_item ${ts_item} \
        --freeze \
        --log \
        --user_sim_func kd \
        --alpha 0.1 \
        --gamma 0.01 \
        --use_cross_att \
        --use_fuzzy
done

# ================================= 2. Bert4Rec 模型（可选，仅模糊约束） =================================
# model_name="llmesr_bert4rec"
# mask_prob=0.6
# # 2.2 LLM-ESR+模糊约束（仅运行此部分）
# for seed in ${seed_list[@]}
# do
#         python main.py --dataset ${dataset} \
#                 --model_name ${model_name} \
#                 --hidden_size 64 \
#                 --train_batch_size 128 \
#                 --max_len 200 \
#                 --gpu_id ${gpu_id} \
#                 --num_workers 8 \
#                 --mask_prob ${mask_prob} \
#                 --num_train_epochs 200 \
#                 --seed ${seed} \
#                 --check_path "fuzzy" \
#                 --patience 20 \
#                 --ts_user ${ts_user} \
#                 --ts_item ${ts_item} \
#                 --freeze \
#                 --log \
#                 --user_sim_func kd \
#                 --alpha 0.1 \
#                 --gamma 0.05 \
#                 --use_cross_att \
#                 --use_fuzzy
# done
