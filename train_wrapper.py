import subprocess
import threading

GPU_ID = [2,3,4]
N = 1
MODEL_NAME = 'distilroberta-base' #'bert-base-uncased' 'distilroberta-base'
#CMD_IDX =  [0,1,..]
seed = [1500, 1390, 5915]

#for i in range(N):
#    cmds = {
#        1: f"python3 train.py --gpu_id={GPU_ID} --seed={seed[i]} --model={MODEL_NAME}"

#    }
#    for j in CMD_IDX:
#        subprocess.call(cmds[j], shell=True)


def run_script(script_name):
    subprocess.call(script_name, shell=True)

if __name__ == "__main__":
    
    script1_thread = threading.Thread(target=run_script, args=(f"python3 train.py --gpu_id={GPU_ID[1]} --seed={seed[1]} --model={MODEL_NAME}",))
    script2_thread = threading.Thread(target=run_script, args=(f"python3 train.py --gpu_id={GPU_ID[2]} --seed={seed[2]} --model={MODEL_NAME}",))
    script3_thread = threading.Thread(target=run_script, args=(f"python3 train.py --gpu_id={GPU_ID[0]} --seed={seed[0]} --model={MODEL_NAME}",))

    script1_thread.start()
    script2_thread.start()
    script3_thread.start()

    script1_thread.join()
    script2_thread.join()
    script3_thread.join()

    print("All scripts have finished executing.")