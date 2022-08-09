#!/bin/bash

#SBATCH --clusters=brain
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=0

#SBATCH --partition=gpu_48g.q
#SBATCH --nodelist=dive601
#SBATCH --job-name=train_det50
#SBATCH --output=train_det50_.out
#SBATCH --error=train_det50_.err

echo "--------"
echo "HOSTNAME = ${HOSTNAME}"
echo "SLURM_JOB_NAME = ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "--------"


# CUDA_VISIBLE_DEVICES=0 
# python tools/train.py configs/__custom/detectors_82split.py
# python tools/train.py configs/__custom/scnet_82split.py
#python tools/train.py configs/__custom/copypaste_82split.py
# python tools/train.py configs/__custom/maskrcnn.py
# python tools/train.py configs/__custom/solov2_82split.py
# python tools/train.py configs/__custom/mask2former_82split.py
# python tools/train.py configs/__custom/detectors_copypaste_82split.py



# # # #CUDA_VISIBLE_DEVICES=3 
#python tools/test.py configs/__custom/detectors_82split.py work_dirs/detectors/2022년_7월_25일_18시_17분_38초_detectors50_100:0_bri0.2_cont0.8_sat0.2_mstrain0.1_scale0.1_AllAug/epoch_${target}.pth \
#--format-only --eval-options jsonfile_prefix=./result_json/detectors50_bri0.2_cont0.8_sat0.2_mstrain0.1_scale0.1_AllAug_epoch_${target}

target=$1
cuda=$2

CUDA_VISIBLE_DEVICES=${cuda} python tools/test.py configs/__custom/detectors_82split.py work_dirs/detectors/2022년_7월_29일_10시_32분_20초_detectors50_trainallTrue_bri0.2_cont0.8_sat0.2_mstrain-0.3_scale0.1_cut400_hole2_prob0.6_epoch30_blur5_LB_Best/epoch_${target}.pth \
--format-only --eval-options jsonfile_prefix=./result_json/detectors50_LBBest_re_epoch_${target} --show-dir ./result/final17
 





#CUDA_VISIBLE_DEVICES=5 
# python tools/test.py configs/__custom/scnet_82split.py work_dirs/scnet/2022년_7월_20일_15시_20분_16초_scnet101_SGD_8:2_colorjitter0.7/epoch_20.pth \
# --format-only --eval-options jsonfile_prefix=./result_json/scnet101_8:2_epoch20_rpnsoftnms0.5 # --show-dir ./result/detectors_82split_100:0_rot360_epoch18_conf0.28_ValPrediction



# EPOCH=$

# python tools/test.py configs/__custom/detectors_82split.py work_dirs/detectors/2022년_7월_10일_13시_44분_10초_detectors50_82split_100:0/epoch_18.pth --format-only --eval-options jsonfile_prefix=./result_json/detectors_82split_100:0_LBBest_epoch_18_maskthr0.42_rpnsoftnms



# LB Best
# python tools/test.py configs/__custom/detectors_82split.py work_dirs/detectors/2022년_7월_10일_13시_44분_10초_detectors50_82split_100:0/epoch_18.pth \
# --format-only --eval-options jsonfile_prefix=./result_json/detectors_82split_100:0_epoch18_LBbest_softnms0.25