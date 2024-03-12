## Steps for running on sagemaker

### 1. Build docker
```
# authenticate
docker login nvcr.io
# username is $oauthtoken
# password is NGC API Key
```
Then build docker
```
cd final_docker
bash build_docker.sh
```

### 2. Tag and push to ECR
```
cd ../
bash push_ecr.sh
```

### 3. Download the model from NGC
Might have to install ngc-cli
```
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.39.0/zip -O ngc_cli_3.39.0.zip
unzip ngc_cli_3.39.0.zip
unzip ngccli_linux.zip
chmod u+x ngc-cli
export PATH=$PATH:ngc-cli
```

Set up ngc config
```
ngc config set
```
Might have to select `nvidian/nemo-llm` as the org and team

Here we download LLama-2-7B engine for 1-A100 GPU
```
ngc registry model download-version "nvidian/nemo-llm/llama-2-7b-chat:LLAMA-2-7B-CHAT-4K-FP16-1-A100.24.02.rc2
```

### 4. Run through the notebook to create SM endpoint
Run `nim_sm_prebuilt_a100.ipynb` notebook as SageMaker notebook instance on `g5.xlarge` instance
