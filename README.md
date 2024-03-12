## Steps for deploying NIM on sagemaker

Launch SageMaker notebook instance on `g5.xlarge` instance. For the sagemaker notebook instance you can use any instance type.
**IMPORTANT:** In Additional Configuration for Volume Size in GB specify at least 300 GB.

### 1. Build/Pull docker container
```
# authenticate
docker login nvcr.io
# username is $oauthtoken
# password is <NGC API Key>
```
Then build docker
```
cd final_docker
bash build_docker.sh nim-24.02-sm-final
```

### 2. Tag and push to ECR
```
cd ../
bash push_ecr.sh nim-24.02-sm-final
```

### 3. Download the prebuilt optimized model from NGC
Install ngc-cli
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

This is expected output. You might have to select `nvidian/nemo-llm` as the org and team
```
(base) [ec2-user@ip-172-16-120-240 ~]$ ngc config set
CLI_VERSION: Latest - 3.40.0 available (current: 3.39.0). Please update by using the command 'ngc version upgrade' 
Enter API key [no-apikey]. Choices: [<VALID_APIKEY>, 'no-apikey']: <NGC_API_KEY>
Enter CLI output format type [ascii]. Choices: ['ascii', 'csv', 'json']: 
Enter org [no-org]: nvidian
Enter team [no-team] : nemo-llm
Enter ace [no-ace] : [no-ace] 
Validating configuration...
Successfully validated configuration.
Saving configuration...
Successfully saved NGC configuration to /home/ec2-user/.ngc/config
```

To see the [list of all available prebuilt models](https://developer.nvidia.com/docs/nemo-microservices/inference/models.html) run the following command. Currently, **NIM supports prebuild models for A100 GPUs (`p4d.24xlarge` instance on AWS)**. Models available on NGC are TensorRT-LLM engine files which are optimized for a specific GPU type and container image version `(YY.MM)`
```
ngc registry model list "nvidian/nemo-llm/*"
```

To make it easy to copy engine name you are interested in you can use
```
ngc registry model list "nvidian/nemo-llm/*" --format_type csv
```

Then use `ngc registry model download-version` to download the prebuilt engine you are interested in. This is the expected format for the command
```
ngc registry model download-version "{Repository}:{Latest Version}"
```

where you can find the `Repository` and `Latest Version` of the model from `ngc registry model list` command

Below we show how to download LLama-2-7B engine which was prebuilt and optimized for running on single A100 GPU, here `Repository="nvidian/nemo-llm/llama-2-7b-chat"`, `Latest Version="LLAMA-2-7B-CHAT-4K-FP16-1-A100.24.02.rc2"`
```
ngc registry model download-version "nvidian/nemo-llm/llama-2-7b-chat:LLAMA-2-7B-CHAT-4K-FP16-1-A100.24.02.rc2
```

### 4. Run through the notebook to deploy on SageMaker
Run `nim_sm_prebuilt_a100.ipynb` notebook to create SageMaker endpoint and deploy the model.
