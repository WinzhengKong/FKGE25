gpu=1
log_folder="../Log_BR_RN/""$(date +%Y%m%d)"

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
learningrate=(1e-4)
byzantine=(RandomNoise)
for km in "${kge_method[@]}"
do
for lr in "${learningrate[@]}"
do
for by in "${byzantine[@]}"
do
cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python -u controller_byzantine.py --cuda \
--local_file_dir ../Data/FB15k-237/R10FL \
--save_dir ../Output/FB15k-237/R10FedEBR"${km}" \
--fed_mode FedE \
--byzantine "${by}" \
--agg FedAD \
--model "${km}" \
--client_num 10 \
--max_epoch 3 \
--max_iter 200 \
--learning_rate "${lr}" \
--hidden_dim 256 \
--valid_iter 5 \
--early_stop_iter 10 \
--cpu_num 16 \
--adm ECOD \
--component1 normalization \
| tee -a "${log_folder}"/R10FedEBR_"${km}"_lr"${lr}"_"${by}"_"${cur_time}".txt
sleep 8
done
done
done