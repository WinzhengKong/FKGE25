gpu=1
log_folder="../Log/FedLU/local/""$(date +%Y%m%d)"
ulimit -n 8192
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
for km in "${kge_method[@]}"
do
cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python -u local.py --cuda \
--local_file_dir ../Data/FB15k-237/R5FL \
--save_dir ../Output/FB15k-237/R5Local \
--model "${km}" \
--agg weighted \
--client_num 5 \
--aggregate_iteration 1 \
--max_epoch 200 \
--valid_epoch 10 \
--hidden_dim 256 \
--early_stop_epoch 5 \
--learning_rate 1e-4 \
--batch_size 512 \
--test_batch_size 16 \
| tee -a "${log_folder}"/R5Local_"${km}"_"${cur_time}".txt
sleep 8
done