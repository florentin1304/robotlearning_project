# python3 train.py --domain udr --delta 1 # --reward_threshold 1500
# python3 train.py --domain udr --delta 5 # --reward_threshold 1500
# python3 train.py --domain udr --delta 10 # --reward_threshold 1500
# python3 train.py --domain udr --delta 0.1 --perc # --reward_threshold 1500
# python3 train.py --domain udr --delta 0.5 --perc # --reward_threshold 1500
# python3 train.py --domain udr --delta 0.95 --perc # --reward_threshold 1500



# python3 test.py --domain target --model models/ppo_udr_1-0_1-0_1-0/best_model --n_episodes 1000
# python3 test.py --domain target --model models/ppo_udr_5-0_5-0_5-0/best_model --n_episodes 1000
# python3 test.py --domain target --model models/ppo_udr_10-0_10-0_10-0/best_model --n_episodes 1000
# python3 test.py --domain target --model models/ppo_udr_0-1_0-1_0-1_perc/best_model --n_episodes 1000
# python3 test.py --domain target --model models/ppo_udr_0-5_0-5_0-5_perc/best_model --n_episodes 1000
# python3 test.py --domain target --model models/ppo_udr_0-95_0-95_0-95_perc/best_model --n_episodes 1000


python3 test.py --domain udr --delta 1 --model models/ppo_udr_1-0_1-0_1-0/best_model --n_episodes 1000
python3 test.py --domain udr --delta 5 --model models/ppo_udr_5-0_5-0_5-0/best_model --n_episodes 1000
python3 test.py --domain udr --delta 10 --model models/ppo_udr_10-0_10-0_10-0/best_model --n_episodes 1000
python3 test.py --domain udr --delta 0.1 --perc --model models/ppo_udr_0-1_0-1_0-1_perc/best_model --n_episodes 1000
python3 test.py --domain udr --delta 0.5 --perc --model models/ppo_udr_0-5_0-5_0-5_perc/best_model --n_episodes 1000
python3 test.py --domain udr --delta 0.95 --perc --model models/ppo_udr_0-95_0-95_0-95_perc/best_model --n_episodes 1000
