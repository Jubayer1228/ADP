# Collect data
#python3 collect_data.py --env linear_bandit --envs 1000 --H 100 --dim 5 --var 0.3 --cov 0.0 --envs_eval 200

# Train
#python3 train.py --env linear_bandit --envs 1000 --H 100 --dim 5 --var 0.3 --cov 0.0 --lr 0.0001 --layer 2 --head 2 --num_epochs 10 --seed 1

# Evaluate, choose an appropriate epoch
python3 eval.py --env bandit --envs 1000 --H 100 --dim 5 --var 0.3 --cov 0.0 --lr 0.0001 --layer 2 --head 2 --epoch 10 --n_eval 200 --seed 1
