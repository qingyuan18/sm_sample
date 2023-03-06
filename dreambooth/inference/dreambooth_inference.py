import os
import json
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import boto3
import sagemaker
import uuid
import torch
from torch import autocast
from PIL import Image
import io
import requests
import traceback
import os
import json
import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import tarfile
import deepspeed 



s3_client = boto3.client('s3')


def measure_latency(pipe, prompt):
    latencies = []
    # warm up
    pipe.set_progress_bar_config(disable=True)
    for _ in range(2):
        _ =  pipe(prompt)
    # Timed run
    for _ in range(10):
        start_time = perf_counter()
        _ = pipe(prompt)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_s = np.mean(latencies)
    time_std_s = np.std(latencies)
    time_p95_s = np.percentile(latencies,95)
    return f"P95 latency (seconds) - {time_p95_s:.2f}; Average latency (seconds) - {time_avg_s:.2f} +\- {time_std_s:.2f};", time_p95_s


def download_model(model_path,target_path):
    bucket, key = get_bucket_and_key(model_path)
    s3_client.download_file(
            bucket,
            key,
            target_path
    )

def get_bucket_and_key(s3uri):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]
    return bucket, key

def model_fn(model_dir):
    """
    Load the model for inference
    """    
    model_name = os.environ['model_name']
    model_path = os.environ['model_path']
    try:
      save_path = '/opt/ml/input/fineturned_model/'
      print("from model path to download:", model_path)
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      #download trained model from model_path(s3uri)
      download_model(model_path,save_path+"model.tar.gz")
      print("model download to target path:", save_path) 
      #extract model.tar.gz in /opt/ml/input/fineturned_model/ folder
      model_file = tarfile.open(save_path+"model.tar.gz")  
      model_file.extractall(save_path)
      print('model extracted: ', os.listdir(save_path))
      model_file.close()
    except Exception as e:
        traceback.print_exc()
        print("download excpetion!")
        print(e)

    
    model_dir='/opt/ml/input/fineturned_model/'
    model = StableDiffusionPipeline.from_pretrained(
        model_dir,
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_dir, subfolder="scheduler"),
        torch_dtype=torch.float16,
        )
    print("model loaded:",model)
 
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    try:
        print("begin load deepspeed....")
        deepspeed.init_inference(
            model=getattr(model,"model", model),      # Transformers models
            mp_size=1,        # Number of GPU
            dtype=torch.float16, # dtype of the weights (fp16)
            replace_method="auto", # Lets DS autmatically identify the layer to replace
            replace_with_kernel_inject=False, # replace the model with the kernel injector
        )
        print("model accelarate with deepspeed!")
    except Exception as e:
        print("deepspeed accelarate excpetion!")
        print(e)


    model = model.to("cuda")
    #model.enable_attention_slicing()

    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    input_data = json.loads(request_body)
    input_data = input_data['inputs']
    
    return input_data

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    print('input_data: ', input_data)
    prompt=input_data['prompt']
    
    try:
        sagemaker_session = sagemaker.Session()
        bucket = sagemaker_session.default_bucket()
        output_s3uri = 's3://{0}/{1}/invoke/images/'.format(bucket, 'stablediffusion_dreambooth')
        repetitions = os.environ['repetitions'] if('repetitions' in os.environ) else 6
        print('repetitions: ', repetitions)
        prediction = []

        with autocast("cuda"):
             images = model(prompt, num_images_per_prompt=repetitions, num_inference_steps=25, guidance_scale=9).images
             for image in images:
                bucket, key = get_bucket_and_key(output_s3uri)
                key = '{0}{1}.jpg'.format(key, uuid.uuid4())
                buf = io.BytesIO()
                image.save(buf, format='JPEG')
                s3_client.put_object(
                    Body = buf.getvalue(), 
                    Bucket = bucket, 
                    Key = key, 
                    ContentType = 'image/jpeg'
                )
                print('image: ', 's3://{0}/{1}'.format(bucket, key))
                prediction.append('s3://{0}/{1}'.format(bucket, key))
    except Exception as e:
        traceback.print_exc()
        print(e)    
    print('prediction: ', prediction)
    return prediction

def output_fn(prediction, content_type):
    """
    Serialize and prepare the prediction output
    """

    return json.dumps(
        {
            'result': prediction
        }
    )