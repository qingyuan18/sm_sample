FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.0-cu117
RUN pip install --no-cache-dir boto3==1.26.64
RUN pip install transformers==4.27.1
RUN pip install icetk
RUN pip install cpm_kernels
RUN pip install uvicorn==0.18.1

RUN pip install --no-cache-dir sagemaker

RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_Linux-64bit.tar.gz | tar -xz && mv s5cmd /opt/conda/bin/

ENV PATH="/opt/ml/code/:${PATH}"
COPY code /opt/ml/code/
WORKDIR /opt/ml/code/
RUN chmod 755 serve


