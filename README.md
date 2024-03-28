# sagemaker-deploy-llms

Amazon SageMaker를 이용하여 HuggingFace의 Large Language Models(LLMs)을 배포하고, 선택적으로 Lambda 함수와 API Gateway를 배포하는 프로젝트입니다.

## 프로젝트 구성

본 프로젝트는 AWS의 여러 서비스를 활용하여 HuggingFace 모델을 배포하는 Python 스크립트를 포함하고 있습니다. 사용자는 스크립트를 통해 쉽게 모델을 SageMaker에 배포하고, 필요에 따라 스트리밍 기능을 활성화할 수 있습니다.

## 사용 방법

### 전제 조건

- AWS 계정과 해당 계정의 `aws-cli`에 대한 접근 권한이 설정되어 있어야 합니다.
- Python 3.x가 설치되어 있어야 합니다.
- 필요한 Python 라이브러리를 설치합니다: `boto3`, `sagemaker`
  ```bash
  pip install boto3 sagemaker
  ```

스크립트는 다음과 같은 명령줄 인자를 받습니다:

- `--s3-uri <model_uri>`: 배포할 HuggingFace 모델의 S3 URI입니다. (필수)
- `--enable-stream`: 스트리밍 기능을 활성화합니다. 이 옵션을 사용하면 SageMaker Endpoint만 배포됩니다.
- `--instance-type`: SageMaker 엔드포인트에 사용할 인스턴스 유형입니다. 기본값은 `ml.g5.12xlarge`입니다.
- `--region`: AWS 리전을 지정합니다. 기본값은 `us-east-1`입니다.
- `--execution-role`: The AWS IAM execution role for SageMaker.


### SageMaker Endpoint 배포
스트리밍 기능을 활성화하면 SageMaer Endpoint만만 배포되며, 스트리밍을 통해 모델에 접근할 수 있습니다.

예제 명령:

```
git clone https://github.com/didhd/sagemaker-deploy-llms
cd sagemaker-deploy-llms
python deploy_script.py --s3-uri s3://your-model-path/model.tar.gz --enable-stream

## 테스트
python3 stream.py HuggingFace-LMM-1710396377
```


### SageMaker Endpoint + Lambda 함수 + API Gateway 배포
`--enable-stream` 옵션을 사용하지 않으면 스크립트는 자동으로 API Gateway까지 설정하여 외부에서 모델에 접근할 수 있도록 합니다.

예제 명령:

```
git clone https://github.com/didhd/sagemaker-deploy-llms
cd sagemaker-deploy-llms
python deploy_script.py --s3-uri s3://your-model-path/model.tar.gz

## 테스트
curl -X POST https://4lb6otx8d0.execute-api.us-east-1.amazonaws.com/prod/invoke \
-H "Content-Type: application/json" \
-d '{
    "inputs": "<s>[INST] <<SYS>> You are a chat bot who writes songs <</SYS>>\n\nWrite a rap song about Amazon Web Services [/INST]</s>",
    "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6},
    "CustomAttributes": "accept_eula=true"
}'
```

## Security

This application was written for demonstration and educational purposes and not for production use. The Security Pillar of the [AWS Well-Architected Framework](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html) can support you in further adopting the sample into a production deployment in addition to your own established processes.


Contributing
------------

1.  Fork the project.
2.  Create your feature branch ( `git checkout -b feature/YourFeature`).
3.  Commit your changes ( `git commit -am 'Add some feature'`).
4.  Push to the branch ( `git push origin feature/YourFeature`).
5.  Open a pull request.

License
-------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.