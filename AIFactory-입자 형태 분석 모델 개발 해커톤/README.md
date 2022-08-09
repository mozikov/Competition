# 입자 형태 분석 모델 개발 해커톤 - Private 4등 (3중대 3소대장)


### 0. Abstract
- MMDetection Library 사용
- model : detectors50
- 그 외 training details은 ```configs/__custom/detectors_82split.py``` 확인
- test시 weight파일은 hnefa335@gmail.com 로 메일.

### 1. Dependency
- torch : 1.12.0+cu113
- torchvision : 0.13.0+cu113
- mmcv-full : 1.5.3
- mmdet : 2.25.0


### 2. Train

python tools/train.py configs/__custom/detectors_82split.py

### 3. Test

python tools/test.py configs/__custom/detectors_82split.py work_dirs/detectors/2022년_7월_29일_10시_32분_20초_detectors50_trainallTrue_bri0.2_cont0.8_sat0.2_mstrain-0.3_scale0.1_cut400_hole2_prob0.6_epoch30_blur5_LB_Best/epoch_${target}.pth \
--format-only --eval-options jsonfile_prefix=./result_json/detectors50_LBBest_re_epoch_15