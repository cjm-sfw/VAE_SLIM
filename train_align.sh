python train_align_with_args.py \
  --vae1_path sd-legacy/stable-diffusion-v1-5 \
  --vae2_path black-forest-labs/FLUX.1-dev \
  --model_version variational \
  --input_types image latent \
  --train_data_dir train_images \
  --eval_data_dir eval_images \
  --image_size 128 \
  --train_batch_size 72 \
  --eval_batch_size 16 \
  --loss_type huber lpips \
  --loss_weight 1.0 0.1 \
  --training_stage 1 \
  --grad_clip 8 \
  --num_workers 4 \
  --it_or_epochs iterations \
  --iterations 100000 \
  --scheduler_step_size 40000 \
  --scheduler_gamma 0.2 \
  --learning_rate 1e-4 \
  --warmup_steps 50 \
  --eval_frequency 10000 \
  --visualize_frequency 1000 \
  --save_frequency 30000

#   --load_checkpoint /workspace/VAE_SLIM/ckpt_align/align_pipeline_20260109_004958.pth \

# python train_align_with_args.py \
#   --vae1_path sd-legacy/stable-diffusion-v1-5 \
#   --vae2_path black-forest-labs/FLUX.1-dev \
#   --input_types image latent \
#   --load_checkpoint /workspace/VAE_SLIM/ckpt_align/align_pipeline_20260109_004958.pth \
#   --train_data_dir train_images \
#   --eval_data_dir eval_images \
#   --train_batch_size 20 \
#   --eval_batch_size 16 \
#   --loss_type mse lpips \
#   --loss_weight 1.0 0.3 \
#   --training_stage 2 \
#   --grad_clip 5 \
#   --num_workers 4 \
#   --epochs 3 \
#   --scheduler_step_size 1 \
#   --scheduler_gamma 0.3 \
#   --learning_rate 1e-4