# VAE_SLIM

## Install

pip install -r requirements.txt
apt-get install libgl1
## env var

cp .env_temp .env

## Dataset

python dataloader.py

## BASE

python train_align_with_args.py \
  --vae1_path sd-legacy/stable-diffusion-v1-5 \
  --vae2_path black-forest-labs/FLUX.1-dev \
  --input_types image latent \
  --load_checkpoint /workspace/VAE_SLIM/ckpt_align/align_pipeline_20260109_004958.pth \
  --train_data_dir train_images \
  --eval_data_dir eval_images \
  --train_batch_size 20 \
  --eval_batch_size 16 \
  --loss_type mse lpips \
  --loss_weight 1.0 0.3 \
  --training_stage 2 \
  --grad_clip 5 \
  --num_workers 4 \
  --epochs 3 \
  --scheduler_step_size 1 \
  --scheduler_gamma 0.3 \
  --learning_rate 1e-4

python eval_align.py \
    --checkpoint /workspace/VAE_SLIM/ckpt_align/align_pipeline_20260109_004958.pth \
    --image_normalize \
    --model_version base \
    --eval_data_dir eval_images \
    --output_dir eval_results \
    --eval_batch_size 1 \
    --num_workers 1 \
    --num_samples 15 \
    --metrics all \
    --sample_mode mean \
    --batch_evaluation \
    --compute_rfid \
    --max_fid_batches 15 \
    --save_visualizations