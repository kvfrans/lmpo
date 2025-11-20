from localutils.tpu_utils import queue_job, run_job
job_list = []

# queue_job('/mount/code/dqlm/lmpo/', 'python core/grpo.py --env_name countdown --num_generation_tokens 1024 --inference_batch_per_device 128 --ppo_minibatch 64 --wandb_group GRPOIter --wandb_name GRPO-v5e')
# run_job('/mount/code/dqlm/lmpo/', 'python core/grpo.py --env_name countdown --num_generation_tokens 256 --inference_batch_per_device 4 --ppo_minibatch 64 --wandb_group GRPOIter --wandb_name GRPO-v5e --groups_per_batch 8', 'v5-node0')

# run_job('/mount/code/dqlm/lmpo/', 'python core/grpo.py --env_name countdown --num_generation_tokens 1024 --inference_batch_per_device 96 --ppo_minibatch 64 --wandb_group Sep10-GRPOIter --wandb_name GRPO-v5e', 'v5-node0')

# job_list = []
# base = 'python core/grpo.py --env_name countdown --num_generation_tokens 1024 --inference_batch_per_device 96 --ppo_minibatch 64 --wandb_group Sep10-GRPOIter'
# job_list.append(base + ' --wandb_name Baseline')
# job_list.append(base + ' --wandb_name AdvClip --do_clip_advantages 1')
# job_list.append(base + ' --wandb_name InferenceRatio --do_inference_logprobs 1')
# job_list.append(base + ' --wandb_name LessWeightDecay --weight_decay 0.001')
# job_list.append(base + ' --wandb_name 64Group --groups_per_batch 64')
# job_list.append(base + ' --wandb_name 16Group --groups_per_batch 16')
# job_list.append(base + ' --wandb_name HighLR --lr 3e-6')
# job_list.append(base + ' --wandb_name LowLR --lr 3e-7')
# job_list.append(base + ' --wandb_name NoEntropyCoef --entropy_coef 0')
# job_list.append(base + ' --wandb_name PPOAllClip --do_ppo_all_clip 1')
# job_list.append(base + ' --wandb_name MaskInferenceRatio --do_mask_inference_ratio 1')
# job_list.append(base + ' --wandb_name LowClip --clip_epsilon 0.05')

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

# base = 'export HF_DATASETS_CACHE=/home/kvfrans/hf_cache; python core/grpo.py --wandb_group Oct23-OfflineDataCollect --entropy_coef 0 --do_mask_inference_ratio 1 --wandb_name Baseline '
# job_list.append(base + ' --env_name countdown --num_generation_tokens 1024 --inference_batch_per_device 96 --ppo_minibatch 64 --save_rollouts_dir /gcs/data/offlinelmpo/countdown/ --do_save_rollouts 1 --groups_per_batch 128')
# job_list.append(base + ' --env_name countdown6 --num_generation_tokens 2048 --inference_batch_per_device 64 --ppo_minibatch 32 --save_rollouts_dir /gcs/data/offlinelmpo/countdown6-v2/ --do_save_rollouts 1 --groups_per_batch 128 --negative_advantage_multiplier 0.6')
# job_list.append(base + ' --env_name deepscaler --num_generation_tokens 2048 --inference_batch_per_device 64 --ppo_minibatch 32 --save_rollouts_dir /gcs/data/offlinelmpo/deepscaler-v2/ --do_save_rollouts 1 --groups_per_batch 128 --prompt_length 512 --save_dir /gcs/checkpoints/offlinelmpo/deepscaler-v2 --save_interval 20')

# base = 'export HF_DATASETS_CACHE=/home/kvfrans/hf_cache; python core/grpo.py --wandb_group Oct31-Debug --entropy_coef 0 --do_mask_inference_ratio 1 --wandb_name Baseline '
# job_list.append(base + ' --env_name countdown --num_generation_tokens 1024 --inference_batch_per_device 32 --ppo_minibatch 64 --groups_per_batch 128')

# for job in job_list:
#     queue_job('/mount/code/dqlm/lmpo/', job)

# run_job('/mount/code/dqlm/lmpo/notes/', 'python profiling.py --ici_dp_parallelism 16 --ici_fsdp_parallelism 1', 'v4-node3') 
# run_job('/mount/code/dqlm/lmpo/notes/', 'python profiling.py --ici_dp_parallelism 1 --ici_fsdp_parallelism 16 --ici_tensor_parallelism 1', 'v4-node2') 


# base = 'python core/grpo.py --env_name countdown --num_generation_tokens 1024 --inference_batch_per_device 64 --wandb_group Nov6-Profiling --entropy_coef 0 --do_mask_inference_ratio 1 --wandb_project lmpo --do_group_filter 0 --groups_per_batch 128'

# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Profiling-FlashXLATrainRemat256 --use_remat 1 --ppo_minibatch 256')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node2') 

# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Profiling-FlashXLATrainRemat64 --use_remat 1 --ppo_minibatch 64')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node1') 

# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Profiling-FlashXLATrainNoRemat64 --use_remat 0 --ppo_minibatch 64')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node5')

# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Profiling-FlashRemat64 --use_remat 1 --ppo_minibatch 64 --use_xla_flags 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node3') 

# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Profiling-FlashRemat128 --use_remat 1 --ppo_minibatch 128 --use_xla_flags 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node1') 

# base = 'python core/grpo.py --env_name countdown --num_generation_tokens 1024 --wandb_group Nov14-ProfilingTraining --entropy_coef 0 --do_mask_inference_ratio 1 --wandb_project lmpo --do_group_filter 0 --groups_per_batch 128'

# job_list = []
# job_list.append(base + ' --wandb_name GRPO-NoFlash-NoFlash-NoXLA-64 --use_flash_attn 0 --use_remat 0 --use_xla_flags 0 --ppo_minibatch 64 --max_steps 1 --profile 1')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node5')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-NoRemat-NoXLA-64 --use_flash_attn 1 --use_remat 0  --use_xla_flags 0 --ppo_minibatch 64 --max_steps 1 --profile 1')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node5')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-NoXLA-64 --use_remat 1 --use_flash_attn 1 --use_xla_flags 0 --ppo_minibatch 64 --max_steps 1 --profile 1')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node5')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-XLA-64 --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 64 --max_steps 1 --profile 1')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node5')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-XLA-256 --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 256 --max_steps 1 --profile 1')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node5')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-NoXLA-256 --use_remat 1 --use_flash_attn 1 --use_xla_flags 0 --ppo_minibatch 256 --max_steps 1 --profile 1')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node5')

# job_list = []
# job_list.append(base + ' --wandb_name GRPO-NoFlash-NoFlash-NoXLA-64 --use_flash_attn 0 --use_remat 0 --use_xla_flags 0 --ppo_minibatch 64 --max_steps 10 --profile 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-NoRemat-NoXLA-64 --use_flash_attn 1 --use_remat 0  --use_xla_flags 0 --ppo_minibatch 64 --max_steps 10 --profile 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-NoXLA-64 --use_remat 1 --use_flash_attn 1 --use_xla_flags 0 --ppo_minibatch 64 --max_steps 10 --profile 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-XLA-64 --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 64 --max_steps 10 --profile 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-XLA-256 --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 256 --max_steps 10 --profile 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-NoXLA-256 --use_remat 1 --use_flash_attn 1 --use_xla_flags 0 --ppo_minibatch 256 --max_steps 10 --profile 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')



# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Remat-Flash-XLA-128 --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 128')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')


base = 'python core/grpo.py --env_name countdown --num_generation_tokens 8192 --wandb_group Nov14-ProfilingTraining --entropy_coef 0 --do_mask_inference_ratio 1 --wandb_project lmpo --do_group_filter 0 --groups_per_batch 16 --inference_batch_per_device 16'
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Remat-Flash-XLA-Len8K-16Batch --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 16')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node0')

# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Remat-Flash-XLA-Len8K-32Batch --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 32')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node1')

job_list = []
job_list.append(base + ' --wandb_name GRPO-Flash-Remat-XLA-8K-32 --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 32 --max_steps 10 --profile 0')
run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')
job_list = []
job_list.append(base + ' --wandb_name GRPO-Flash-Remat-NoXLA-8K-32 --use_remat 1 --use_flash_attn 1 --use_xla_flags 0 --ppo_minibatch 32 --max_steps 10 --profile 0')
run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-XLA-256 --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 256 --max_steps 10 --profile 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')
# job_list = []
# job_list.append(base + ' --wandb_name GRPO-Flash-Remat-XLA-256 --use_remat 1 --use_flash_attn 1 --use_xla_flags 1 --ppo_minibatch 256 --max_steps 10 --profile 0')
# run_job('/mount/code/dqlm/lmpo/', job_list[0], 'v4-node4')