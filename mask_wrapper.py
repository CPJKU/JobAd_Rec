import subprocess
import threading

GPU_ID = [2,3,4]
N = 1
base_path = '/share/rk8/home/deepak/JobAd_Rec/Models/'
Model_paths = ['1500_distilroberta-base_2024-01-25_02-54-56','1390_distilroberta-base_2024-01-25_02-54-56','5915_distilroberta-base_2024-01-25_02-54-56'] 
seed = [1500, 1390, 5915]



def run_script(script_name):
    subprocess.call(script_name, shell=True)

if __name__ == "__main__":
    
    script1_thread = threading.Thread(target=run_script, args=(f"python3 mask.py --gpu_id={GPU_ID[1]} --seed={seed[1]} --model_path={base_path+Model_paths[1]}",))
    script2_thread = threading.Thread(target=run_script, args=(f"python3 mask.py --gpu_id={GPU_ID[2]} --seed={seed[2]} --model_path={base_path+Model_paths[2]}",))
    script3_thread = threading.Thread(target=run_script, args=(f"python3 mask.py --gpu_id={GPU_ID[0]} --seed={seed[0]} --model_path={base_path+Model_paths[0]}",))

    script1_thread.start()
    script2_thread.start()
    script3_thread.start()

    script1_thread.join()
    script2_thread.join()
    script3_thread.join()

    print("All scripts have finished executing.")