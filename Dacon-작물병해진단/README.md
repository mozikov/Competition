# 농업 환경 변화에 따른 작물 병해 진단 AI 경진대회 Private 3rd , 3중대 3소대장


# Flow

### 1. Background
이번 대회 데이터는 **모델이 예측에 혼동을 겪는 데이터가 대다수입니다.**    
이는 5Fold 검증을 해봤을 때 명확해지는데요.   
즉, 제 경험상 분류하기에 명확한 특징이 있는 데이터라면 폴드마다 결국 작은 오차범위내로 convergence가 되지만    
개인적인 실험 결과 각 폴드 내에서 연속되는 epoch 간에 3-6% 내외 큰 오차범위로 흔들리는 모습을 보였습니다.   
(구체적으로 Resnet50 5fold 기준   
1fold -> 87-93%   
2fold -> 89-92%   
3fold -> 88-92%   
...   
각 폴드별300Epoch 훈련. LR 1e-3~1e-5까지 down, RAdam Optimizer, Step scheduler)   

이 말이 시사하는 바는 **Public Data 뿐만 아니라 Private Data까지 이와같은 현상이 일어날 것이며**   
**즉, 더욱 더 Generalization된 Model을 만들어야 한다라고 결론을 내렸습니다.**   

<br />

### 2. Method

그렇다면 위에서 말한 Generalization을 위해 어떠한 선택을 해야할 것인가가 문제입니다.   
이를 위해 저는 다음과 같은 방법을 취하였습니다.

#### 2-1. Ensemble
Generalization을 위해서는 다양한 모델의 Ensemble이 필수라고 생각했습니다.   
5Fold를 하지 않고 다양한 모델의 Ensemble을 선택했습니다.   
어찌보면 매우 단순해 보이지만 대회에서 시간적 제약에 따른 점수 차감도 있기 때문에 5Fold 로는 다양한 모델을 선택할 수 없다고 생각했습니다.   
**그렇기에 5Fold 보다는 총 7개의 Model을 Ensemble 하였습니다.**     



#### 2-2. Augmentation 
**분명하게 성능을 하락시키지만 않는다면 그 Augmentation은 무조건 선택하는 방법을 취했습니다.**   
이 과정에서도 약간의 혼동이 있었습니다.   
예를 들어 ColorJitter Augmentation을 했을때, 이게 성능이 올라가진 않지만 그렇다고 떨어뜨리는것 같진 않고.. 그냥 제자리 걸음인가?   
하는 의문점이 계속 들었습니다. 왜냐하면 앞서 말했듯이 Validation F1 score가 계속 흔들렸기 때문이죠.   
이렇게 애매한 경우에는 무조건 포함시켰습니다.   
왜냐하면 Validation Set에서는 성능 향상이 뚜렷하진 않지만, 훨씬 개수가 많은 Test Data에서는 긍정적 작용을 할 수도 있고,   
그렇기에 Test Data에 Generalization하는데에 도움이 될 수 있을거라 생각했기 때문입니다.   

**최종 선택한 Augmentation은 다음과 같습니다.**   
ColorJitter(0.1), Cutout(min20, max40), HVFlip, Affine(shift0.2, scale0.2), BoxCrop   

이중에서 Affine은 대회 후반부가 되어서야 적용하면 좋겠다고 생각들어 후반부 모델들에 적용하였으며,   
Shift, Scale을 0.2씩 준 것은 적용 시 정보 손실이 크지 않고   
이미지 내 작물이 정 가운데에 있지 않으며 각각의 작물마다 상하좌우 위치가 다르기에 선택했습니다.      

BoxCrop이란 제가 커스텀으로 적용한 방법이며,   
주어진 작물의 box 좌표를 이용하여 그 부분만(또는 가로 세로 비율 조금 더 늘려서) 이미지를 Crop하였습니다.   

#### 2-3. Data Split
Training 99, Validation 1 의 Data Split 방법을 취했습니다.   
구체적으로는 Stratified하게 100Fold로 나누었습니다.   
이때 Seed는 고정한 채, 2개의 모델씩 짝을 지어 하나는 1번째 fold를 validation,   
나머지 하나는 100번째 fold를 validation하는 방법을 취했습니다.    


#### 2-4. etc
Label Smoothing - 대회 후반부에 실험했던 모델에 Label Smoothing(0.1)을 적용하였습니다.   
Focal Loss - 마찬가지로 대회 후반부 모델에 Focal Loss를 적용하였습니다.   
Cutmix - 대회 후반부에 3Fold로 검증해봤을때 확실히 효과가 있다고 판단하여 적용한 모델을 만들어 Ensemble 했지만   
생각보다 Public Data에서 점수가 하락하여 고민끝에 제외하였습니다.
시간적 여유가 없어서 추가적인 실험을 못했지만, 아마 초기 모델부터 Cutmix를 적절히 적용하여 더 실험했으면 성능이 좋아졌을거라 생각합니다.     

<br />

### 3. Training Details

Weight : 모든 모델은 ImageNet에서 pretrained된 weight를 초깃값으로 사용   
Optimizer : RAdam(1e-3)   
Scheduler : Step Scheduler(step_size=100, 0.1)   
Epochs : 300   

Model   

|            |Model|Augmentation|ETC|
|------------|-----------|-----------|--------------|
1            | Resnet50 | Colorjitter(0.1), Cutout(min20, max40), H, V flip, rotate90 | Cross Entropy Loss + TTAx6(H, Vflip, rotate90, 180, 270)
2            | Resnet18 | Colorjitter(0.1), Cutout(min20, max40), H, V flip, Affine, BoxCrop | Focal Loss + TTAx3(H, Vflip)
3            | EfficientNet-b3 | Colorjitter(0.1), Cutout(min20, max40), H, V flip, Affine, BoxCrop | Focal Loss + TTAx3(H, Vflip)
4            | eca_vovnet39b | Colorjitter(0.1), Cutout(min20, max40), H, V flip, rotate90, BoxCrop | Cross Entropy Loss + TTAx3(H, Vflip)
5            | Densenet201 | Colorjitter(0.1), Cutout(min20, max40), H, V flip, rotate90, BoxCrop |  Cross Entropy Loss + TTAx3(H, Vflip)
6            | res2net50_26w_4s | Colorjitter(0.1), Cutout(min20, max40), H, V flip, Affine, BoxCrop |  Cross Entropy Loss(Label_Smoothing=0.1) + TTAx3(H, Vflip)
7            | resnext50_32x4d | Colorjitter(0.1), Cutout(min20, max40), H, V flip, Affine, BoxCrop |  Cross Entropy Loss(Label_Smoothing=0.1) + TTAx3(H, Vflip)
