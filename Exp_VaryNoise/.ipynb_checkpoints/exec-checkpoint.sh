CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_experiment.py --DATA_SET kddcup --num_neg_samples 2 --max_gamma 1
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 run_experiment.py --DATA_SET kddcup_neptune --num_neg_samples 2 --max_gamma 1
CUDA_VISIBLE_DEVICES=2,3,0,1 python3 run_experiment.py --DATA_SET nb15 --num_neg_samples 2 --max_gamma 1
CUDA_VISIBLE_DEVICES=3,0,1,2 python3 run_experiment.py --DATA_SET nsl_kdd --num_neg_samples 2 --max_gamma 1
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 run_experiment.py --DATA_SET gureKDD --num_neg_samples 2 --max_gamma 1