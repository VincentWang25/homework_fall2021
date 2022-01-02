source ~/.bashrc
# Q1 sub-q 1
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q1_env1_random
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --exp_name q1_env2_random

# Q2 sub-q 1
# cql_alpha = 0 => DQN, cql_alpha = 0.1 => CQL
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1


# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql_shift1scale100 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 1 --exploit_rew_scale 100

# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql_shift0scale10 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 0 --exploit_rew_scale 10

# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql_shift10scale1 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 10 --exploit_rew_scale 1

# Q2 sub-q 2
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_5000 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_15000 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_5000 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_15000 

# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_5000 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_15000 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_5000 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_15000 

# # Q2 sub-q 3
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.02 --exp_name q2_alpha0.02 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.5 --exp_name q2_alpha0.5 

# # Q3
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_medium_dqn 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_medium_cql 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_hard_dqn 
# python3 cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_hard_cql 

# Q4
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q5_awac_medium_unsupervised_lam0.1 --use_rnd --unsupervised_exploration --awac_lambda=0.1 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q5_awac_medium_unsupervised_lam1 --use_rnd --unsupervised_exploration --awac_lambda=1 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q5_awac_medium_unsupervised_lam2 --use_rnd --unsupervised_exploration --awac_lambda=2 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q5_awac_medium_unsupervised_lam10 --use_rnd --unsupervised_exploration --awac_lambda=10 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q5_awac_medium_unsupervised_lam20 --use_rnd --unsupervised_exploration --awac_lambda=20 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q5_awac_medium_unsupervised_lam50 --use_rnd --unsupervised_exploration --awac_lambda=50 --num_exploration_steps=20000

python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=0.1 --exp_name q5_awac_medium_supervised_lam0.1
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=1 --exp_name q5_awac_medium_supervised_lam1
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=2 --exp_name q5_awac_medium_supervised_lam2
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --exp_name q5_awac_medium_supervised_lam10
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=20 --exp_name q5_awac_medium_supervised_lam20
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --exp_name q5_awac_medium_supervised_lam50

python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q5_awac_easy_unsupervised_lam0.1 --use_rnd --unsupervised_exploration --awac_lambda=0.1 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q5_awac_easy_unsupervised_lam1 --use_rnd --unsupervised_exploration --awac_lambda=1 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q5_awac_easy_unsupervised_lam2 --use_rnd --unsupervised_exploration --awac_lambda=2 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q5_awac_easy_unsupervised_lam10 --use_rnd --unsupervised_exploration --awac_lambda=10 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q5_awac_easy_unsupervised_lam20 --use_rnd --unsupervised_exploration --awac_lambda=20 --num_exploration_steps=20000
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q5_awac_easy_unsupervised_lam50 --use_rnd --unsupervised_exploration --awac_lambda=50 --num_exploration_steps=20000

python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=0.1 --exp_name q5_awac_easy_supervised_lam0.1
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=1 --exp_name q5_awac_easy_supervised_lam1
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=2 --exp_name q5_awac_easy_supervised_lam2
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --exp_name q5_awac_easy_supervised_lam10
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=20 --exp_name q5_awac_easy_supervised_lam20
python3 cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --exp_name q5_awac_easy_supervised_lam50
