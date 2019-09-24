export CUDA_VISIBLE_DEVICES=5
nohup python -u translate.py -model experiments/checkpoints2/model_step_500000.pt -gpu 0 -src data2/src-test.txt -output pred.txt -beam_size 5 -n_best 5 -batch_size 128 -replace_unk -max_length 200 -fast > trans.out &
