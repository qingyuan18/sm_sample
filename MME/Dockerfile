FROM 802834080501.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-tritonserver:22.10-py3
#install any additional libraries
RUN echo "hello"
RUN mkdir -p /home/condpackenv/

# Copy conda env
COPY sd_env.tar.gz /home/condpackenv/sd_env.tar.gz
