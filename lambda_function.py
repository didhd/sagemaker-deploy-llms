import os
import boto3
import json

# Initialize the SageMaker client
sagemaker_runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    # Extract the SageMaker endpoint name from environment variables
    endpoint_name = os.environ['ENDPOINT_NAME']
    
    # Log the received event for debugging purposes
    print("Received event:", json.dumps(event))
    
    # Extract the 'body' from the event and load it as JSON
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        # Return a 400 error if there is an issue with JSON decoding
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON in request body"})
        }
    
    # Invoke the SageMaker endpoint with the body as the input
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(body)  # Forward the parsed 'body' as input to the SageMaker model
        )
        
        # Read the response from SageMaker endpoint
        result = response['Body'].read().decode('utf-8')
        
        # Return the model's response
        return {
            "statusCode": 200,
            "body": result
        }
    except Exception as e:
        print("Error invoking SageMaker endpoint:", e)
        # Return a 500 error if there is an issue invoking the endpoint
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Error invoking SageMaker endpoint"})
        }
