# screen -S eval0 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run0_worker0.config" --out_file="run_outs/run0_worker0" --llm_gpu="cuda:0" --rm_gpu="cuda:1"; bash'
# screen -S eval1 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run0_worker1.config" --out_file="run_outs/run0_worker1" --llm_gpu="cuda:2" --rm_gpu="cuda:3"; bash'
# screen -S eval2 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run0_worker2.config" --out_file="run_outs/run0_worker2" --llm_gpu="cuda:4" --rm_gpu="cuda:5"; bash'
# screen -S eval3 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run0_worker3.config" --out_file="run_outs/run0_worker3" --llm_gpu="cuda:6" --rm_gpu="cuda:7"; bash'

#screen -S eval0 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run0b_worker0.config" --out_file="run_outs/run0b_worker0" --llm_gpu="cuda:0" --rm_gpu="cuda:1"; bash'

#screen -S eval1 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run0b_worker1.config" --out_file="run_outs/run0b_worker1" --llm_gpu="cuda:2" --rm_gpu="cuda:3"; bash'

# screen -S eval0 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run1_worker0.config" --out_file="run_outs/run1_worker0" --llm_gpu="cuda:0" --rm_gpu="cuda:1"; bash'
# screen -S eval1 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run1_worker1.config" --out_file="run_outs/run1_worker1" --llm_gpu="cuda:2" --rm_gpu="cuda:3"; bash'
# screen -S eval2 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run1_worker2.config" --out_file="run_outs/run1_worker2" --llm_gpu="cuda:4" --rm_gpu="cuda:5"; bash'
# screen -S eval3 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run1_worker3.config" --out_file="run_outs/run1_worker3" --llm_gpu="cuda:6" --rm_gpu="cuda:7"; bash'
screen -S eval0 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run2b_worker0.config" --out_file="run_outs/run2b_worker0" --llm_gpu="cuda:0" --rm_gpu="cuda:1" 2>&1 | tee logs/worker0.log; bash'
screen -S eval1 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run2b_worker1.config" --out_file="run_outs/run2b_worker1" --llm_gpu="cuda:2" --rm_gpu="cuda:3" 2>&1 | tee logs/worker1.log; bash'
screen -S eval2 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run2b_worker2.config" --out_file="run_outs/run2b_worker2" --llm_gpu="cuda:4" --rm_gpu="cuda:5" 2>&1 | tee logs/worker2.log; bash'
screen -S eval3 -dm bash -c 'python3 collect_model_outs.py --run_percent 25 --config="run_configs/run2b_worker3.config" --out_file="run_outs/run2b_worker3" --llm_gpu="cuda:6" --rm_gpu="cuda:7" 2>&1 | tee logs/worker3.log; bash'
