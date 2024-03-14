import boto3
import sagemaker
import json
import argparse
import io


def get_realtime_response_stream(sagemaker_runtime, endpoint_name, payload):
    response_stream = sagemaker_runtime.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name,
        Body=json.dumps(payload),
        ContentType="application/json",
        CustomAttributes="accept_eula=true",
    )
    return response_stream


def print_response_stream(response_stream):
    event_stream = response_stream["Body"]
    start_json = b"{"
    stop_token = "</s>"
    for line in LineIterator(event_stream):
        if line != b"" and start_json in line:
            data = json.loads(line[line.find(start_json) :].decode("utf-8"))
            if data["token"]["text"] != stop_token:
                print(data["token"]["text"], end="")


class LineIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


def build_llama2_prompt(instructions):
    stop_token = "</s>"
    start_token = "<s>"
    startPrompt = f"{start_token}[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, instruction in enumerate(instructions):
        if instruction["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{instruction['content']}\n<</SYS>>\n\n")
        elif instruction["role"] == "user":
            conversation.append(instruction["content"].strip())
        else:
            conversation.append(
                f"{endPrompt} {instruction['content'].strip()} {stop_token}{startPrompt}"
            )

    return startPrompt + "".join(conversation) + endPrompt


def get_instructions(user_content):
    """
    Note: We are creating a fresh user content everytime by initializing instructions for every user_content.
    This is to avoid past user_content when you are inferencing multiple times with new ask everytime.
    """

    system_content = """
    You are a friendly and knowledgeable email marketing agent, Mr.MightyMark, working at AnyCompany. 
    Your goal is to send email to subscribers to help them understand the value of the new product and generate excitement for the launch.

    Here are some tips on how to achieve your goal:

    Be personal. Address each subscriber by name and use a friendly and conversational tone.
    Be informative. Explain the key features and benefits of the new product in a clear and concise way.
    Be persuasive. Highlight how the new product can solve the subscriber's problems or improve their lives.
    Be engaging. Use emojis to make your emails more visually appealing and interesting to read.

    By following these tips, you can use email marketing to help your company launch a successful software product.
    """

    instructions = [
        {"role": "system", "content": f"{system_content} "},
    ]

    instructions.append({"role": "user", "content": f"{user_content}"})
    return instructions


def main(endpoint_name):
    boto3.setup_default_session(region_name="us-east-1")
    sagemaker_runtime = boto3.client("sagemaker-runtime")

    user_ask_1 = """
    AnyCompany recently announced new service launch named AnyCloud Internet Service.
    Write a short email about the product launch with Call to action to Alice Smith, whose email is alice.smith@example.com
    Mention the Coupon Code: EARLYB1RD to get 20% for 1st 3 months.
    """
    instructions = get_instructions(user_ask_1)
    prompt = build_llama2_prompt(instructions)
    print(prompt)
    inference_params = {
        "do_sample": True,
        "top_p": 0.7,
        "temperature": 1,
        "top_k": 50,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.03,
        "stop": ["</s>"],
        "return_full_text": False,
    }

    payload = {
        "inputs": prompt,
        "parameters": inference_params,
        "stream": True,  ## <-- to have response stream.
    }

    resp = get_realtime_response_stream(sagemaker_runtime, endpoint_name, payload)
    print_response_stream(resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream response from a specified SageMaker endpoint."
    )
    parser.add_argument(
        "endpoint_name", type=str, help="The name of the SageMaker endpoint to query."
    )
    args = parser.parse_args()

    main(args.endpoint_name)
