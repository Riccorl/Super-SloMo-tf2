echo "Inference"
python super-slomo/generate_evaluation.py ucf101_slomo \
  "/content/drive/My Drive/Sapienza/Magistrale/Computer Vision/models/run11/chckpnt/ckpt-259"

echo 'Evaluating'
python super-slomo/eval_video_interpolation.py \
	--gt-dir ucf101_interp_ours \
	--motion-mask-dir motion_masks_ucf101_interp/ \
	--res-dir ucf101_superslomo_adobe240fps/ \
	--res-suffix _01_interp.png
