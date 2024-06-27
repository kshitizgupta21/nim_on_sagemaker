# Steps for deploying NVIDIA NIM on sagemaker

## Launch SageMaker notebook instance

Launch SageMaker notebook instance on `g5.xlarge` instance and git clone this repo inside that instance.

### 1. Pull and Build SageMaker compatible NIM docker container
In terminal of SageMaker notebook instance run the following commands. First we authenticate into NGC
```
# authenticate into NGC
docker login nvcr.io
# username is $oauthtoken
# password is <NGC API Key>
```
Then pull docker for specific LLM NIM
```
export SRC_IMAGE_PATH=nvcr.io/nim/meta/llama3-8b-instruct:latest

export SRC_IMAGE_NAME="${SRC_IMAGE_PATH##*/}"
export SRC_IMAGE_NAME="${SRC_IMAGE_NAME%%:*}"
docker pull ${SRC_IMAGE_PATH}
docker tag ${SRC_IMAGE_PATH} ${SRC_IMAGE_NAME}

sed 's/{{ SRC_IMAGE }}/$SRC_IMAGE_PATH/g' Dockerfile > Dockerfile.tmp
envsubst < Dockerfile.tmp > Dockerfile.nim
docker build -f Dockerfile.nim -t ${SRC_IMAGE_NAME} .
```

### 2. Tag and push container to ECR
```
bash push_ecr.sh ${SRC_IMAGE_NAME}
```

### 3. If you want to deploy LLama-3 8B model on g5.xlarge (A10G GPU) on SageMaker
Run `nim_llama3_8b_a10g.ipynb` notebook 


