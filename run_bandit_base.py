# Collect data
#python3 collect_data.py --env bandit --envs 30000 --H 200 --dim 5 --var 0.3 --cov 0.0 --envs_eval 200

# Train
#python3 train.py --env bandit --envs 30000 --H 200 --dim 5 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --shuffle --num_epochs 25 --seed 1

# Evaluate, choose an appropriate epoch
python3 eval.py --env bandit --envs 30000 --H 200 --dim 5 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --shuffle --epoch 25 --n_eval 200 --seed 1
