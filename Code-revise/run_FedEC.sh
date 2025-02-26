gpu=0
log_folder="../Log_FedEC/""$(date +%Y%m%d)"

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
aggregate_method=(weighted)
learningrate=(1e-4)
mucontrastive=(0.1)
mutemperature=(0.2)
for km in "${kge_method[@]}"
do
for lr in "${learningrate[@]}"
do
for mc in "${mucontrastive[@]}"
do
for mt in "${mutemperature[@]}"
do
cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python -u controller_byzantine.py --cuda \
--local_file_dir ../Data/FB15k-237/R10FL \
--save_dir ../Output/FB15k-237/R10FedEC"${km}" \
--fed_mode FedEC \
--agg weighted \
--model "${km}" \
--client_num 10 \
--max_epoch 3 \
--max_iter 200 \
--hidden_dim 256 \
--learning_rate "${lr}" \
--valid_iter 5 \
--early_stop_iter 5 \
--test_batch_size 16 \
--mu_contrastive "${mc}" \
--mu_temperature "${mt}" \
--cpu_num 16 \
| tee -a "${log_folder}"/R10FedEC_"${km}"_con"${mc}"_tem"${mt}"_"${am}"_dist"${dm}"_lr"${lr}"_"${cur_time}".txt
sleep 8
done
done
done
done