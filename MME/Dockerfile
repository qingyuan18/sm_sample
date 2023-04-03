FROM 301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:22.10-py3
#install any additional libraries
RUN echo "hello"
RUN mkdir -p /home/condpackenv/

# Copy conda env
COPY sd_env.tar.gz /home/condpackenv/sd_env.tar.gz
