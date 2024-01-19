import os
import time

cuda_list = [0]
seed_list = [0]

algo_list = [
    "atppo", 
    "ppo",
    "trpo",
    "sac",
    "td3",
    ]

task_list = [
    "ShadowHandOver", 
    "ShadowHandCatchUnderarm",
    "ShadowHandOver2Underarm",
    "ShadowHandCatchAbreast",
    "ShadowHandTwoCatchUnderarm",
    "ShadowHandPushBlock",
    "ShadowHandScissors",
    "ShadowHandBlockStack",
    "ShadowHandPourWater",
    "ShadowHandGraspAndPlace",
    "ShadowHandLiftUnderarm",
    "ShadowHandDoorCloseInward",
    "ShadowHandDoorOpenInward",
    "ShadowHandDoorCloseOutward",
    "ShadowHandDoorOpenOutward",
    "ShadowHandPen",
    "ShadowHandSwingCup",
    "ShadowHandReOrientation",
    "ShadowHandSwitch",
    "ShadowHandBottleCap",
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