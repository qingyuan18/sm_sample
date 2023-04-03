import boto3
import traceback
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import tarfile
import subprocess


s3_client = boto3.client('s3')

def download_model(model_path,target_path):
    bucket, key = get_bucket_and_key(model_path)
    s3_client.download_file(
        bucket,
        key,
        target_path
    )

def convert_to_sd_model(src_model_path,is_safetensor,dst_model_path):
    command = "python ./convert_original_stable_diffusion_to_diffusers.py"
    # python convert_original_stable_diffusion_to_diffusers.py  --checkpoint_path ../../dreamshaper_331BakedVae.safetensors --from_safetensors  --to_safetensors --dump_path ../../dreamshaper
    command = command +"--checkpoint_path "+ src_format
    if is_safetensor:
        command = command + " --from_safetensors"
    command= command + " --dump_path "+dst_model_path
    try:
        process = subprocess.Popen(command, shell=True)
        process.wait()
        print("convert format finished!")
    except Exception as e:
        traceback.print_exc()
        print("convert format excpetion!")
        print(e)



def get_bucket_and_key(s3uri):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]
    return bucket, key




def load_stable_diffusion_model(model_path):
    base_name=os.path.basename(model_path)
    model_dir = '/opt/ml/input/fineturned_model/'
    try:
        if "s3" in model_path:
            print("from model path to download:", model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            #download trained model from model_path(s3uri)
            download_model(model_path,model_dir+"model.tar.gz")
            print("model download to target path:", model_dir)

        if "model.tar.gz" == base_name:
            #extract model.tar.gz in /opt/ml/input/fineturned_model/ folder
            model_file = tarfile.open(model_dir+"model.tar.gz")
            model_file.extractall(model_dir)
            print('model extracted: ', os.listdir(model_dir))
            model_file.close()
        elif "safetensors" == base_name:
            convert_to_sd_model(src_model_path=model_dir,is_safetensor=True,dst_model_path=model_dir)
        elif "ckpt" == base_name:
            convert_to_sd_model(src_model_path=model_dir,is_safetensor=False,dst_model_path=model_dir)
        else:
            print("not support source model format:"+base_name)
    except Exception as e:
      traceback.print_exc()
      print(e)

    model = StableDiffusionPipeline.from_pretrained(
        model_dir,
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_dir, subfolder="scheduler"),
        torch_dtype=torch.float16
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("model loaded:",model)

    model = model.to("cuda")
    model.enable_attention_slicing()
    return model