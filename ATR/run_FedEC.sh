gpu=1
log_folder="../Log/FedEC/""$(date +%Y%m%d)"

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

kge_method=(ComplEx DisMult)
learningrate=(1e-4)
byzantine=(None)
maliciousratio=(0.4)
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
for by in "${byzantine[@]}"
do
for mr in "${maliciousratio[@]}"
do
cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python -u controller_byzantine.py --cuda \
--local_file_dir ../Data/FB15k-237/R10FL \
--save_dir ../Output/FB15k-237/R10FedECB"${km}" \
--fed_mode FedEC \
--agg weighted \
--model "${km}" \
--client_num 10 \
--max_epoch 3 \
--max_iter 200 \
--byzantine "${by}" \
--hidden_dim 256 \
--learning_rate "${lr}" \
--valid_iter 5 \
--early_stop_iter 50 \
--test_batch_size 16 \
--mu_contrastive "${mc}" \
--mu_temperature "${mt}" \
--cpu_num 16 \
--malicious_ratio "${mr}" \
| tee -a "${log_folder}"/R10FedECB_"${km}"_con"${mc}"_tem"${mt}"_dist"${dm}"_lr"${lr}"_"${by}"_"${mr}"_"${cur_time}".txt
sleep 8
done
done
done
done
done
done