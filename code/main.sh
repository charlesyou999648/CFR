# to generate the wrong predict embeddings:
python generate.py

# or you go ahead and use the already generated ones.

python run_expt.py --dropout=0.1 --one_divide=True --dataset waterbirds --algorithm CFR --model clip-rn50 --root_dir data --device 0 --freeze_language --freeze_vision --use_wandb --train_projection --seed 222222222 --batch_size 128 --n_epochs 100 --class_weight 0 --clip_weight 1.0 --image_weight 1.0 --language_weight 1.0 --domain_weight 0.0 --spurious_weight 0.0 --spurious_class_weight 1.0 --spurious_clip_weight 0.0 --crossmodal_weight 0.0 --pos_weight 1.0 --neg_weight 1.0 --weight_decay 1e-5 --lr 1e-4 --download=True 
python run_expt.py --dataset_kwargs model=vit --dropout=0.1 --one_divide=True --dataset waterbirds --algorithm CFR --model clip-vit --root_dir data --device 0 --freeze_language --freeze_vision --use_wandb --train_projection --seed 222222222 --batch_size 128 --n_epochs 100 --class_weight 0 --clip_weight 1.0 --image_weight 1.0 --language_weight 1.0 --domain_weight 0.0 --spurious_weight 0.0 --spurious_class_weight 1.0 --spurious_clip_weight 0.0 --crossmodal_weight 0.0 --pos_weight 1.0 --neg_weight 1.0 --weight_decay 1e-5 --lr 1e-4 --download=True 
