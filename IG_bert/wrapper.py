import subprocess

GPU_ID = 3

jobs = ['software engineer',
    #'senior software engineer',
    #'dentist',
    #'accountant',
    #'architect',                 
    #'teacher',
    #'nurse',
    #'paralegal',
    #'painter',
    #'psychologist',
    #'interior designer',
    #'photographer',
    #'dietitian',
    #'personal trainer'
    ]
CMD_IDX =  [0]

for i in jobs:

    cmds = {
        0: f"python3 /home/deepak/RecSys2023/IG_bert/words_score.py --gpu_id={GPU_ID} --job={repr(i)}"
    }
    for j in CMD_IDX:
        subprocess.call(cmds[j], shell=True)