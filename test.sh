export CUDA_VISIBLE_DEVICES=0
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
#   --data_root_dir <PATH TO OXE DATA ROOT> \
  --data_root_dir datasets/open-x-embodiment \
  --run_root_dir checkpoints \
#   --wandb_project "<PROJECT>" \
#   --wandb_entity "<ENTITY>"
echo "test dataset"
