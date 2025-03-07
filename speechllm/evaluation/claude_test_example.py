# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# snippet-start:[python.example_code.bedrock-runtime.Converse_AnthropicClaude]
# Use the Conversation API to send a text message to Anthropic Claude.

import boto3
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Set the model ID, e.g., Claude 3 Haiku.
# model_id = "anthropic.claude-3-haiku-20240307-v1:0"
# model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

def generate_claude_response(input_message, model_id:str = "anthropic.claude-3-sonnet-20240229-v1:0", max_tokens:int = 512, temperature:float = 0.2, top_p = 0.8):
    # Start a conversation with the user message.
    # input_message = "Describe the purpose of a 'hello world' program in one line."
    
    conversation = [
        {
            "role": "user",
            "content": [{"text": input_message}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": top_p},
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)
        return response_text 

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)
