aws ecr create-repository --repository-name "nim-24.02-sm-final" > /dev/null

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 354625738399.dkr.ecr.us-east-1.amazonaws.com

docker tag nim-24.02-sm-final 354625738399.dkr.ecr.us-east-1.amazonaws.com/nim-24.02-sm-final

docker push 354625738399.dkr.ecr.us-east-1.amazonaws.com/nim-24.02-sm-final