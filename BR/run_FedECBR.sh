gpu=1
log_folder="../Log_BR/FedEC/""$(date +%Y%m%d)"

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

kge_method=(RotatE)
byzantine=(Poison)
learningrate=(1e-4)
mucontrastive=(0.1)
mutemperature=(0.2)
admodel=(ECOD)
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
for ad in "${admodel[@]}"
do
cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python -u controller_byzantine.py --cuda \
--local_file_dir ../Data/FB15k-237/R10FLPN \
--save_dir ../Output/FB15k-237/R10FedECBR"${km}" \
--fed_mode FedEC \
--agg FedAD \
--model "${km}" \
--client_num 10 \
--max_epoch 3 \
--max_iter 200 \
--hidden_dim 256 \
--learning_rate "${lr}" \
--valid_iter 5 \
--early_stop_iter 50 \
--test_batch_size 16 \
--mu_contrastive "${mc}" \
--mu_temperature "${mt}" \
--byzantine "${by}" \
--cpu_num 16 \
--adm "${ad}" \
| tee -a "${log_folder}"/R10FedECBR_"${km}"_con"${mc}"_tem"${mt}"_"${am}"_dist"${dm}"_lr"${lr}"_"${by}"_adm"${ad}"_"${cur_time}".txt
sleep 8
done
done
done
done
done
done