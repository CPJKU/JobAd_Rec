import subprocess
import threading

GPU_ID = [4,5]
N = 1
MODEL_NAME = 'bert-base-uncased' #'bert-base-uncased' 'distilroberta-base'
seed = [1500, 1390, 5915]
lmbda = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
db = ["reg"]



def run_script(script_name):
    subprocess.call(script_name, shell=True)

if __name__ == "__main__":
    for s in seed:
        for l in lmbda:
            script1_thread = threading.Thread(target=run_script, args=(f"python3 train.py --gpu_id={GPU_ID[0]} --seed={s} --debias='reg' --lmbda={l} --model={MODEL_NAME}",))
            # script2_thread = threading.Thread(target=run_script, args=(f"python3 train.py --gpu_id={GPU_ID[2]} --seed={seed[2]} --model={MODEL_NAME}",))
            # script3_thread = threading.Thread(target=run_script, args=(f"python3 train.py --gpu_id={GPU_ID[0]} --seed={seed[0]} --model={MODEL_NAME}",))
            #
            script1_thread.start()
            # script2_thread.start()
            # script3_thread.start()
            #
            # script1_thread.join()
            # script2_thread.join()
            script1_thread.join()

    print("All scripts have finished executing.")