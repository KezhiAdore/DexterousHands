import os
import time

cuda_list = [0,1,2]
seed_list = [0,1,2]

algo_list = [
    "atppo", 
    "ppo",
    "trpo",
    "sac",
    "td3",
    ]

task_list = [
    "ShadowHandOver", 
    "ShadowHandScissors"
    "ShadowHandPen",
    "ShadowHandSwingCup",
    "ShadowHandDoorCloseInward",
    "ShadowHandDoorOpenInward",
    "ShadowHandDoorCloseOutward",
    "ShadowHandDoorOpenOutward",
    "ShadowHandReOrientation"
    ]

count = 0
for task in task_list:
    for algo in algo_list:
        for seed in seed_list:
            cuda_id = cuda_list[count%len(cuda_list)]
            cmd = f"nohup python train.py \
                    --task={task} \
                    --algo={algo} \
                    --rl_device=cuda:{cuda_id} \
                    --sim_device=cuda:{cuda_id} \
                    --seed={seed} \
                    --headless \
                    2>&1 >/dev/null &"
            count+=1
            print(cmd)
            os.system(cmd)
            time.sleep(2)

# # 统计个数
# ps -ef |grep -v grep|grep headless|wc -l

# # 终止所有进程
# ps -aux|grep headless |grep -v grep|awk '{print "kill -9 "$2}'|sh