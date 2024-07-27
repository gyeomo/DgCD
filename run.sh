data='PACS'
# 'PACS'
# 'VLCS'
# 'OfficeHome'
# 'TerraIncognita'
# 'DomainNet'

rho=0.05
ratio=0.33
for t in `seq 0 2`
do
    python train_all.py $data \
    --trial_seed $t \
    --data_dir ./dataset \
    --algorithm ERM_DgCD \
    --dataset $data \
    --lr 1e-5 \
    --resnet_dropout 0.0 \
    --weight_decay 1e-6 \
    --rho $rho \
    --ratio $ratio
done
