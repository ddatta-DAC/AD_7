CUDA_VISIBLE_DEVICES=0,2,3,1 python3 run_experiment.py --DATA_SET kddcup --num_runs 2
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 run_experiment.py --DATA_SET kddcup_neptune --num_runs 2
CUDA_VISIBLE_DEVICES=2,3,1,0 python3 run_experiment.py --DATA_SET gureKDD --num_runs 2
CUDA_VISIBLE_DEVICES=3,2,0,1 python3 run_experiment.py --DATA_SET nsl_kdd --num_runs 2
CUDA_VISIBLE_DEVICES=3,0,2,1 python3 run_experiment.py --DATA_SET nb15 --num_runs 2