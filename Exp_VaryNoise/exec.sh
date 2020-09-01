CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_experiment.py --DATA_SET kddcup --num_neg_samples 2 
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 run_experiment.py --DATA_SET kddcup_neptune --num_neg_samples 2
CUDA_VISIBLE_DEVICES=1,0,2,3 python3 run_experiment.py --DATA_SET nb15 --num_neg_samples 2 
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_experiment.py --DATA_SET nsl_kdd --num_neg_samples 2 
CUDA_VISIBLE_DEVICES=1,1,2,3 python3 run_experiment.py --DATA_SET gureKDD --num_neg_samples 2 