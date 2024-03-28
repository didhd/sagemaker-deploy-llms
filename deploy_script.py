import argparse
import boto3
import time
import sys
import logging
import subprocess
import os
import json


logging.getLogger("sagemaker.config").setLevel(logging.WARNING)

from sagemaker.huggingface import HuggingFaceModel
from sagemaker.huggingface import get_huggingface_llm_image_uri

import sagemaker

endpoint_name = f"HuggingFace-LMM-{int(time.time())}"


def find_sagemaker_execution_role():
    iam_client = boto3.client("iam")
    roles = iam_client.list_roles()
    for role in roles["Roles"]:
        if role["RoleName"].startswith("AmazonSageMaker-ExecutionRole"):
            return role["Arn"]
    raise Exception(
        "No SageMaker execution role found that starts with 'AmazonSageMaker-ExecutionRole', please use --execution-role <execution_role_arn>"
    )


def get_gpu_count(instance_type):
    # 인스턴스 타입과 GPU 수의 매핑
    instance_gpu_mapping = {
        "ml.p2.xlarge": 1,
        "ml.p2.8xlarge": 8,
        "ml.p2.16xlarge": 16,
        "ml.p3.2xlarge": 1,
        "ml.p3.8xlarge": 4,
        "ml.p3.16xlarge": 8,
        "ml.g4dn.xlarge": 1,
        "ml.g4dn.12xlarge": 4,
        "ml.g5.xlarge": 1,
        "ml.g5.2xlarge": 1,
        "ml.g5.4xlarge": 1,
        "ml.g5.8xlarge": 1,
        "ml.g5.16xlarge": 1,
        "ml.g5.12xlarge": 4,
        "ml.g5.24xlarge": 4,
        "ml.g5.48xlarge": 8,
        "ml.p4d.24xlarge": 8,
        "ml.p4de.24xlarge": 8,
        "ml.p5.48xlarge": 8,
    }

    # 지정된 인스턴스 타입에 대한 GPU 수 반환
    return instance_gpu_mapping.get(
        instance_type, 1
    )  # Default to 1 if instance type is not found


def deploy_sagemaker_endpoint(s3_uri, instance_type, region, execution_role=None):
    """
    Deploys a HuggingFace Large Language Model (LLM) to Amazon SageMaker.

    This function automates the process of deploying a HuggingFace model on SageMaker. It uses the model data stored in an S3 bucket, specified by the s3_uri parameter. The deployment utilizes a specified instance type and targets the specified AWS region.

    Parameters:
    - s3_uri (str): The S3 URI where the HuggingFace model data is stored. The URI should point to a .tar.gz file containing the model.
    - instance_type (str): The type of instance on which to deploy the model. For example, 'ml.g5.12xlarge'.
    - region (str): The AWS region in which to deploy the model.

    Returns:
    - None: The function prints the endpoint name of the deployed model but does not return any value.
    """

    # SageMaker 세션 및 역할 설정
    boto3.setup_default_session(region_name=region)
    sagemaker_session = sagemaker.Session()

    # 실행 역할이 제공되지 않은 경우, 기본 역할을 찾습니다.
    if execution_role is None:
        execution_role = find_sagemaker_execution_role()

    # S3 URI가 .tar.gz 파일을 가리키는지 확인
    if s3_uri.endswith(".tar.gz"):
        model_data = s3_uri
    else:
        # .tar.gz로 끝나지 않으면 S3DataSource 형식을 사용
        model_data = {
            "S3DataSource": {
                "S3Uri": s3_uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        }

    # sagemaker config
    health_check_timeout = 300
    number_of_gpu = get_gpu_count(instance_type=instance_type)

    # Define Model and Endpoint configuration parameter
    config = {
        "HF_MODEL_ID": "/opt/ml/model",  # path to where sagemaker stores the model
        "SM_NUM_GPUS": json.dumps(number_of_gpu),
    }

    # retrieve the llm image uri
    llm_image = get_huggingface_llm_image_uri(
        "huggingface", version="1.4.2", session=sagemaker_session
    )

    huggingface_model = HuggingFaceModel(
        model_data=model_data,
        role=execution_role,
        sagemaker_session=sagemaker_session,
        image_uri=llm_image,
        env=config,
    )

    print(f"Deploying SageMaker endpoint: {endpoint_name}")
    # 모델 배포하고 엔드포인트 이름 가져오기
    llm = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=health_check_timeout,
    )

    sagemaker_client = boto3.client("sagemaker", region_name=region)
    while True:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status == "Creating":
            time.sleep(1)
        elif status == "InService":
            print("Deployment complete! SageMaker endpoint is ready.")

            print("\n. Use the following command to test the SageMaker Endpopint:")
            print(f"python3 stream.py {endpoint_name}\n")
            break
        elif status == "Failed":
            print("Deployment failed. Check the AWS console for more information.")
            sys.exit()
        else:
            time.sleep(1)


def create_zip_file(zip_file_path, source_file_path):
    # ZIP 파일이 존재하는지 확인하고, 있으면 삭제
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)

    # ZIP 파일 생성
    subprocess.check_call(["zip", zip_file_path, source_file_path])


def create_role_with_policies(role_name):
    iam_client = boto3.client("iam")

    # AssumeRole 정책 문서
    assume_role_policy_document = """{
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {
            "Service": "lambda.amazonaws.com"
          },
          "Action": "sts:AssumeRole"
        }
      ]
    }"""

    # IAM 역할 생성
    try:
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=assume_role_policy_document,
        )
        role_arn = role["Role"]["Arn"]
        print(f"Created role: {role_arn}")
    except iam_client.exceptions.EntityAlreadyExistsException:
        print(f"Role {role_name} already exists.")
        role_arn = iam_client.get_role(RoleName=role_name)["Role"]["Arn"]

    # 필요한 정책 연결
    policies = [
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    ]

    for policy_arn in policies:
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        print(f"Attached policy {policy_arn} to role {role_name}")

    return role_arn


def create_lambda_function(endpoint_name, region, zip_file_path, source_file_path):
    """
    Creates an API Gateway for a given Lambda function to allow HTTP access.

    Args:
    lambda_function_name (str): The name of the Lambda function.
    region (str): AWS region where the API Gateway and Lambda function are deployed.

    Returns:
    str: The URL endpoint of the deployed API Gateway.
    """
    lambda_client = boto3.client("lambda", region_name=region)
    iam_client = boto3.client("iam", region_name=region)

    role_name = "LambdaExecutionRoleForSageMakerEndpoint"
    function_name = f"SageMakerFunction-{endpoint_name}"

    # Check for and create IAM role
    try:
        role = iam_client.get_role(RoleName=role_name)
        role_arn = role["Role"]["Arn"]
    except iam_client.exceptions.NoSuchEntityException:
        print(f"The role {role_name} does not exist. Creating the role...")
        role_arn = create_role_with_policies(role_name)

    # Create ZIP file from the lambda.py file
    create_zip_file(zip_file_path, source_file_path)

    # Read the ZIP file content
    with open(zip_file_path, "rb") as f:
        zipped_code = f.read()

    # Check if the Lambda function exists
    try:
        response = lambda_client.get_function(FunctionName=function_name)
        # If the function exists, update its code and configuration, including timeout
        print(f"Updating existing Lambda function: {function_name}")
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zipped_code,
        )
        function_arn = response["Configuration"]["FunctionArn"]
        return function_arn
    except lambda_client.exceptions.ResourceNotFoundException:
        # If the function does not exist, create it with the specified timeout
        print(f"Creating new Lambda function: {function_name}")
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime="python3.8",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zipped_code},
            Environment={"Variables": {"ENDPOINT_NAME": endpoint_name}},
            Timeout=60,  # Set timeout to 60 seconds
        )
        function_arn = response["FunctionArn"]
        print(f"Lambda function created: {function_arn}")
        return function_arn
    except boto3.exceptions.Boto3Error as error:
        print(f"Failed to create or update Lambda function: {error}")
        sys.exit()


def create_api_gateway_for_lambda(lambda_function_name, region):
    api_client = boto3.client("apigateway", region_name=region)
    lambda_client = boto3.client("lambda", region_name=region)
    sts_client = boto3.client("sts", region_name=region)

    # 현재 AWS 계정 ID 가져오기
    account_id = sts_client.get_caller_identity()["Account"]

    # API 생성
    api = api_client.create_rest_api(
        name=f"LambdaSageMakerAPI-{lambda_function_name}",
        description="API for SageMaker Lambda Function",
    )
    api_id = api["id"]
    root_resource_id = api_client.get_resources(restApiId=api_id)["items"][0]["id"]

    # 리소스 및 메서드 생성
    resource = api_client.create_resource(
        restApiId=api_id, parentId=root_resource_id, pathPart="invoke"
    )
    resource_id = resource["id"]

    # POST 메소드 추가
    api_client.put_method(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod="POST",
        authorizationType="NONE",
    )

    # Lambda 함수에 권한 추가
    lambda_arn = lambda_client.get_function(FunctionName=lambda_function_name)[
        "Configuration"
    ]["FunctionArn"]
    source_arn = f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/POST/invoke"

    lambda_client.add_permission(
        FunctionName=lambda_function_name,
        StatementId=f"sagemaker-invoke-permission",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=source_arn,
    )

    # Lambda 함수를 메서드에 통합 (AWS_PROXY 타입 사용)
    uri = f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"
    api_client.put_integration(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod="POST",
        type="AWS_PROXY",
        integrationHttpMethod="POST",
        uri=uri,
    )

    # CORS 설정 추가
    api_client.put_method_response(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod="POST",
        statusCode="200",
        responseModels={"application/json": "Empty"},
        responseParameters={
            "method.response.header.Access-Control-Allow-Headers": False,
            "method.response.header.Access-Control-Allow-Methods": False,
            "method.response.header.Access-Control-Allow-Origin": False,
        },
    )
    api_client.put_integration_response(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod="POST",
        statusCode="200",
        responseTemplates={"application/json": ""},
        responseParameters={
            "method.response.header.Access-Control-Allow-Headers": "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
            "method.response.header.Access-Control-Allow-Methods": "'OPTIONS,POST'",
            "method.response.header.Access-Control-Allow-Origin": "'*'",
        },
    )

    # API 배포
    deployment = api_client.create_deployment(restApiId=api_id, stageName="prod")
    endpoint_url = f"https://{api_id}.execute-api.{region}.amazonaws.com/prod/invoke"
    print(f"API Gateway endpoint URL: {endpoint_url}")
    # Print curl command example for testing the API Gateway endpoint
    curl_command = f"""curl -X POST {endpoint_url} \\
-H "Content-Type: application/json" \\
-d '{{
    "inputs": "<s>[INST] <<SYS>> You are a chat bot who writes songs <</SYS>>\\n\\nWrite a rap song about Amazon Web Services [/INST]</s>",
    "parameters": {{"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6}},
    "CustomAttributes": "accept_eula=true"
}}'
"""
    print("\n2. Use the following curl command to test the API Gateway endpoint:")
    print(curl_command)
    return endpoint_url


def main():
    parser = argparse.ArgumentParser(
        description="Deploy a model to Amazon SageMaker with optional streaming support. If streaming is not enabled, deploys a Lambda function, also sets up API Gateway."
    )
    parser.add_argument(
        "--enable-stream",
        action="store_true",
        help="Enable streaming (deploys SageMaker Endpoint only)",
    )
    parser.add_argument("--s3-uri", type=str, required=True, help="S3 URI of the model")
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.g5.12xlarge",
        help="The instance type on which to deploy the SageMaker model",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region to deploy resources in",
    )
    parser.add_argument(
        "--execution-role", type=str, help="The AWS IAM execution role for SageMaker"
    )

    args = parser.parse_args()

    # 설정된 리전으로 boto3 클라이언트 설정 (예시: SageMaker, Lambda, API Gateway)
    boto3.setup_default_session(region_name=args.region)

    # SageMaker 엔드포인트 배포
    deploy_sagemaker_endpoint(
        args.s3_uri, args.instance_type, args.region, args.execution_role
    )

    # 기존 코드 중 일부를 수정하여 조건을 추가합니다.
    if not args.enable_stream:
        # 스트리밍이 활성화되지 않은 경우, Lambda 배포 후 API Gateway까지 설정
        lambda_function_arn = create_lambda_function(
            endpoint_name, args.region, "lambda.zip", "lambda_function.py"
        )
        create_api_gateway_for_lambda(lambda_function_arn, args.region)


if __name__ == "__main__":
    main()
