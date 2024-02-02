python3 train.py --domain source #--reward_threshold 1500
python3 train.py --domain target #--reward_threshold 1500

python3 test.py --domain source --model models/ppo_source/best_model --n_episodes 1000 # source -> source
python3 test.py --domain target --model models/ppo_source/best_model --n_episodes 1000 # source -> target (lower bound)
python3 test.py --domain target --model models/ppo_target/best_model --n_episodes 1000 # target -> target (upper bound)
