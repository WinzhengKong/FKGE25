gpu=0
log_folder="../Log/FedLU/""$(date +%Y%m%d)"

while getopts "g:l:" opt;
do
    case ${opt} in
        g) gpu=$OPTARG ;;
        l) log_folder=$OPTARG ;;
        *) echo "Invalid option: $OPTARG" ;;
    esac
done

echo "log folder: " "${log_folder}"
if [[ ! -d ${log_folder} ]];then
    mkdir -p "${log_folder}"
    echo "create log folder: " "${log_folder}"
fi

kge_method=(TransE RotatE)
dist_para=(2.0)
gamma=(10.0 20.0 9.0)
learningrate=(1e-4)
for km in "${kge_method[@]}"
do
for dm in "${dist_para[@]}"
do
for g in "${gamma[@]}"
do
for lr in "${learningrate[@]}"
do
cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python -u fedlu.py --cuda \
--local_file_dir ../Data/FB15k-237/R5FL \
--save_dir ../Output/FB15k-237/R5FedLU"${km}" \
--fed_mode FedDist \
--co_dist \
--dist_mu "${dm}" \
--agg weighted \
--model "${km}" \
--client_num 5 \
--max_epoch 3 \
--max_iter 200 \
--hidden_dim 256 \
--learning_rate "${lr}" \
--gamma "${g}" \
--valid_iter 5 \
--early_stop_iter 10 \
--wait_iter 20 \
--test_batch_size 64 \
--batch_size 512 \
--cpu_num 16 \
| tee -a "${log_folder}"/R5FedLU_"${km}"_gamma"${g}"_dist"${dm}"_lr"${lr}"_"${cur_time}".txt
sleep 8
done
done
done
done
