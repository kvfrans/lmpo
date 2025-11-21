from localutils.tpu_utils import queue_job, run_job
job_list = []


# base = 'python core/grpo.py --env_name countdown --num_generation_tokens 1024 --inference_batch_per_device 96 --ppo_minibatch 64 --wandb_group Sep28-Branching --entropy_coef 0 --do_mask_inference_ratio 1 --wandb_project lmpo'
# job_list.append(base + ' --wandb_name GRPO')
# base = 'python core/grpo.py --env_name countdown --num_generation_tokens 1024 --inference_batch_per_device 96 --ppo_minibatch 64 --wandb_group Sep28-Branching --entropy_coef 0 --do_mask_inference_ratio 1 --wandb_project lmpo --do_group_normalization 0 --do_global_normalization 1 --do_group_filter 0 --groups_per_batch 2048 --group_size 1'
# job_list.append(base + ' --wandb_name PPO-NoBaseline')

# base = 'python core/grpo.py --env_name countdown --num_generation_tokens 1024 --inference_batch_per_device 96 --ppo_minibatch 64 --wandb_group Sep11-GRPOIter --entropy_coef 0 --do_mask_inference_ratio 1'
# job_list.append(base + ' --wandb_name DebugTFSRP')
# job_list.append(base + ' --wandb_name LowLR --lr 3e-7')

# base = 'python core/grpo.py --env_name countdown6 --inference_batch_per_device 96 --ppo_minibatch 64 --wandb_group Sep11-GRPOIter --entropy_coef 0 --do_mask_inference_ratio 1 --wandb_group Debug'
# base = 'python core/grpo.py --env_name countdown6 --inference_batch_per_device 96 --ppo_minibatch 64 --wandb_group Sep11-GRPOIter --entropy_coef 0 --do_mask_inference_ratio 1 --wandb_group Debug'
# job_list.append(base + ' --wandb_name Baseline --num_generation_tokens 8192 --groups_per_batch 8 --do_group_filter 0 --sharding tfsdp')
# job_list.append(base + ' --wandb_name Baseline --num_generation_tokens 256 --groups_per_batch 8 --do_group_filter 0')
# job_list.append(base + ' --wandb_name Baseline --num_generation_tokens 256 --groups_per_batch 8 --do_group_filter 0 --sharding tfsdp')
# job_list.append(base + ' --wandb_name Baseline --num_generation_tokens 256 --groups_per_batch 8 --sharding tfsdp')


# base = 'python core/grpo.py --env_name deepscaler --num_generation_tokens 2048 --inference_batch_per_device 32 --ppo_minibatch 32 --wandb_group Sep11-GRPOIter --entropy_coef 0 --do_mask_inference_ratio 1 --prompt_length 512 --wandb_name Baseline-2048'
# job_list.append(base + ' --wandb_name Baseline-2048')
# base = 'python core/grpo.py --env_name deepscaler --num_generation_tokens 2048 --inference_batch_per_device 32 --ppo_minibatch 32 --wandb_group Sep11-GRPOIter --entropy_coef 0 --do_mask_inference_ratio 1 --prompt_length 512'
# job_list.append(base + ' --wandb_name Baseline-2048-128Group --groups_per_batch 128')

# run_job('/mount/code/dqlm/lmpo/', "export HF_DATASETS_CACHE='/home/kvfrans/hf_cache' ; python core/grpo.py --env_name deepscaler --num_generation_tokens 2048 --inference_batch_per_device 32 --ppo_minibatch 32 --wandb_group Sep11-GRPOIter --entropy_coef 0 --do_mask_inference_ratio 1 --prompt_length 512 --wandb_name Baseline-2048", 'v4-node0')
# run_job('/mount/code/dqlm/lmpo/', "export HF_DATASETS_CACHE='/home/kvfrans/hf_cache' ; python core/grpo.py --env_name deepscaler --num_generation_tokens 2048 --inference_batch_per_device 32 --ppo_minibatch 32 --wandb_group Sep11-GRPOIter --entropy_coef 0 --do_mask_inference_ratio 1 --prompt_length 512 --wandb_name Baseline-2048-128Group --groups_per_batch 128", 'v4-node1')

# job_list = []
# job_list.append('python core/eval.py --env_name deepscaler --num_generation_tokens 1024 --inference_batch_per_device 64 --prompt_length 512')
# job_list.append('python core/eval.py --env_name deepscaler --num_generation_tokens 2048 --inference_batch_per_device 32 --prompt_length 512')
# job_list.append('python core/eval.py --env_name deepscaler --num_generation_tokens 4096 --inference_batch_per_device 16 --prompt_length 512')
# job_list.append('python core/eval.py --env_name aime --num_generation_tokens 1024 --inference_batch_per_device 64 --num_epochs 8 --prompt_length 512')
# job_list.append('python core/eval.py --env_name aime --num_generation_tokens 2048 --inference_batch_per_device 32 --num_epochs 8 --prompt_length 512')
# job_list.append('python core/eval.py --env_name aime --num_generation_tokens 4096 --inference_batch_per_device 16 --num_epochs 8 --prompt_length 512')

# base = "export HF_DATASETS_CACHE=/home/kvfrans/hf_cache ; python core/grpo.py --env_name deepscaler --inference_batch_per_device 16 --ppo_minibatch 16 --wandb_group Oct15-DeepscalerBaselines --entropy_coef 0 --do_mask_inference_ratio 1 --prompt_length 512"
# job_list.append(base + ' --wandb_name Baseline-8192-128Group --num_generation_tokens 8192 --groups_per_batch 32')
# job_list.append(base + ' --wandb_name Baseline-2048-128Group --num_generation_tokens 4096 --groups_per_batch 128')
# job_list.append(base + ' --wandb_name Baseline-2048-128Group --num_generation_tokens 8192 --groups_per_batch 128')

# base = 'export HF_DATASETS_CACHE=/home/kvfrans/hf_cache; python core/grpo.py --wandb_group Nov20-OfflineDataCollect --entropy_coef 0 --do_mask_inference_ratio 1 '

# job_list.append(base + ' --env_name countdown6 --num_generation_tokens 1024 --inference_batch_per_device 128 --ppo_minibatch 64 --save_rollouts_dir /gcs/data/offlinelmpo/v3-countdown6-1024/ --do_save_rollouts 1 --groups_per_batch 128 --negative_advantage_multiplier 0.6')
# job_list.append(base + ' --env_name countdown6 --num_generation_tokens 2048 --inference_batch_per_device 64 --ppo_minibatch 64 --save_rollouts_dir /gcs/data/offlinelmpo/v3-countdown6-2048/ --do_save_rollouts 1 --groups_per_batch 64 --negative_advantage_multiplier 0.6')
# job_list.append(base + ' --env_name countdown6 --num_generation_tokens 4096 --inference_batch_per_device 32 --ppo_minibatch 32 --save_rollouts_dir /gcs/data/offlinelmpo/v3-countdown6-4096/ --do_save_rollouts 1 --groups_per_batch 32 --negative_advantage_multiplier 0.6')
# job_list.append(base + ' --env_name countdown6 --num_generation_tokens 8192 --inference_batch_per_device 16 --ppo_minibatch 32 --save_rollouts_dir /gcs/data/offlinelmpo/v3-countdown6-8192/ --do_save_rollouts 1 --groups_per_batch 16 --negative_advantage_multiplier 0.6')

# job_list.append(base + ' --wandb_name Baseline-1024 --env_name deepscaler --num_generation_tokens 1024 --inference_batch_per_device 96 --ppo_minibatch 64 --save_rollouts_dir /gcs/data/offlinelmpo/v3-deepscaler-1024/ --do_save_rollouts 1 --groups_per_batch 96 --negative_advantage_multiplier 0.6  --prompt_length 512 --save_dir /gcs/checkpoints/offlinelmpo/v3-deepscaler-1024/')
# job_list.append(base + ' --wandb_name Baseline-2048 --env_name deepscaler --num_generation_tokens 2048 --inference_batch_per_device 64 --ppo_minibatch 64 --save_rollouts_dir /gcs/data/offlinelmpo/v3-deepscaler-2048/ --do_save_rollouts 1 --groups_per_batch 64 --negative_advantage_multiplier 0.6  --prompt_length 512 --save_dir /gcs/checkpoints/offlinelmpo/v3-deepscaler-2048/')
# job_list.append(base + ' --wandb_name Baseline-4096 --env_name deepscaler --num_generation_tokens 4096 --inference_batch_per_device 32 --ppo_minibatch 32 --save_rollouts_dir /gcs/data/offlinelmpo/v3-deepscaler-4096/ --do_save_rollouts 1 --groups_per_batch 32 --negative_advantage_multiplier 0.6  --prompt_length 512 --save_dir /gcs/checkpoints/offlinelmpo/v3-deepscaler-4096/')
# job_list.append(base + ' --wandb_name Baseline-8192 --env_name deepscaler --num_generation_tokens 8192 --inference_batch_per_device 16 --ppo_minibatch 32 --save_rollouts_dir /gcs/data/offlinelmpo/v3-deepscaler-8192/ --do_save_rollouts 1 --groups_per_batch 16 --negative_advantage_multiplier 0.6  --prompt_length 512 --save_dir /gcs/checkpoints/offlinelmpo/v3-deepscaler-8192/')


base = 'export HF_DATASETS_CACHE=/home/kvfrans/hf_cache; '
# job_list.append(base + 'python core/eval.py --env_name deepscaler --num_generation_tokens 1024 --inference_batch_per_device 32 --prompt_length 512 --model_dir /gcs/checkpoints/offlinelmpo/v3-deepscaler-4096/step80/ --force_subsample 2048')
job_list.append(base + 'python core/eval.py --env_name deepscaler --num_generation_tokens 4096 --inference_batch_per_device 32 --prompt_length 512 --model_dir /gcs/checkpoints/offlinelmpo/v3-deepscaler-1024/step80/ --force_subsample 2048')



for job in job_list:
    queue_job('/mount/code/dqlm/lmpo/', job)