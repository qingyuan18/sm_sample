FROM public.ecr.aws/o7x6j3x6/sd-dreambooth-finetuning-v2

RUN pip install --no-cache-dir boto3==1.26.64

RUN pip install --no-cache-dir fastapi uvicorn sagemaker
RUN pip install --no-cache-dir diffusers==0.14.0 accelerate==0.17.0 controlnet_aux

RUN pip install s3fs==2023.3.0
RUN pip install https://publish.djl.ai/deepspeed/deepspeed-0.8.0-py2.py3-none-any.whl

RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_Linux-64bit.tar.gz | tar -xz && mv s5cmd /opt/conda/bin/

ENV PATH="/opt/ml/code/:${PATH}"
COPY code /opt/ml/code/
WORKDIR /opt/ml/code/
RUN chmod 755 serve


