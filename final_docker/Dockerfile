FROM nvcr.io/nvidian/nemo-llm/nemollm-inference-trt-only:24.02.rc4
# SageMaker expects all containers to run with root users
USER root
# Downgrade pynvml to make it compatible with 470 driver which is currently used by SM A10G, A100
RUN apt update && apt install -y python3-pip
RUN pip install pynvml==11.4.1
# Set an environment variable for the common path
ENV NEMOLLM_PATH=/usr/local/lib/python3.10/dist-packages/nemollm_inference

# Use the environment variable in the COPY instructions
COPY sagemaker.py $NEMOLLM_PATH/sdk/entrypoints/
COPY nemollm_inference_ms.py $NEMOLLM_PATH/service/nemollm_inference_ms.py
COPY entrypoints_from_cli.py $NEMOLLM_PATH/service/entrypoints_from_cli.py 
COPY serve /usr/bin/.
RUN chmod 777 /usr/bin/serve
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
EXPOSE 8080