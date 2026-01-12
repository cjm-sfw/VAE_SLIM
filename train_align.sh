python train_align_with_args.py \
  --vae1_path sd-legacy/stable-diffusion-v1-5 \
  --vae2_path black-forest-labs/FLUX.1-dev \
  --model_version longtail \
  --input_types image latent \
  --train_data_dir train_images \
  --eval_data_dir eval_images \
  --train_batch_size 160 \
  --eval_batch_size 16 \
  --loss_type l1 \
  --loss_weight 1.0 \
  --training_stage 1 \
  --grad_clip 2.5 \
  --num_workers 16 \
  --epochs 3 \
  --scheduler_step_size 1 \
  --scheduler_gamma 0.3 \
  --learning_rate 1e-3

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
#   --loss_type l1 lpips \
#   --loss_weight 1.0 1.0 \
#   --training_stage 2 \
#   --grad_clip 5 \
#   --num_workers 4 \
#   --epochs 3 \
#   --scheduler_step_size 1 \
#   --scheduler_gamma 0.3 \
#   --learning_rate 1e-5