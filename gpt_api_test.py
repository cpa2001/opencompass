# #需要先 pip install openai
import os
from openai import AzureOpenAI
import httpx

client = AzureOpenAI(
    #azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    #api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    # azure_endpoint = "https://gpt-4o-zngd4.openai.azure.com/",
    # azure_endpoint="https://gpt-4-0125-preview-zngd3.openai.azure.com/",
    azure_endpoint="https://gpt-35-turbo-instruct-zn.openai.azure.com/",
    # api_key = "8bc954da91ff4340b63a8c47b57135fc",
    # api_key="f9c6bc1a51184aa296954a54fae66f8f",
    api_key="a55c9d72f7ef48d7b637923d6e8815ed",
    # api_version = "2024-02-01",
    api_version="2024-06-01",
)
# why="你在干什么"
# response = client.completions.create(
#     #model="gpt-35-turbo", # model = "deployment_name".
#     model = "gpt-4-0125-Preview",
#     prompt=why
#         # {"role": "system", "content": "You are a helpful assistant."},
#         # {"role": "user", "content": "azureopenai所用的api_key和直接调用gpt4所用的api_key有什么不同吗"},
#         # {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
#         # {"role": "user", "content": "Do other Azure AI services support this too?"}
    
# )

# # print(response.choices[0].message.content)
# text = response.choices[0].text
# print(why+text)

# import os
# import openai

# openai.api_key = os.getenv("be97ca8fe759430e90770ee99b4b86df")
# openai.api_base = os.getenv("https://gpt-35-turbo-instruct-zngd.openai.azure.com/") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
# openai.api_type = 'azure'
# openai.api_version = '2024-02-15-preview' # this might change in the future

# deployment_name='gpt-35-turbo-instruct' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

# # Send a completion call to generate an answer
# print('Sending a test completion job')
# start_phrase = 'Write a tagline for an ice cream shop. '
# response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=10)
# text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
# print(start_phrase+text)

# #需要先 pip install openai
# import os
# from openai import AzureOpenAI

# client = AzureOpenAI(
#     #azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
#     #api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
#     azure_endpoint = "https://gpt-4-0125-preview-zntd.openai.azure.com/",
#     api_key = "a6295ec40e094fa5bbf62adfbad7617c",
#     api_version = "2024-02-15-preview"
# )

# response = client.chat.completions.create(
#     # model = "deployment_name".
#     # model = "gpt-4-0125-Preview",
#     # model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
#         {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
#         {"role": "user", "content": "Do other Azure AI services support this too?"}
#     ]
# )
# print(response.choices[0].message.content)
try:
    # 对于 gpt-35-turbo-instruct，使用 completions 而不是 chat.completions
    response = client.completions.create(
    # response = client.chat.completions.create(
        model="gpt-35-turbo-instruct",
        prompt="Does Azure OpenAI support customer managed keys?", 
        max_tokens=100,
    #     messages=[
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
    #     {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
    #     {"role": "user", "content": "Do other Azure AI services support this too?"}
    # ]
    )
    print(response.choices[0].text)
except Exception as e:
    print(f"An error occurred: {e}")
    if isinstance(e, httpx.ProxyError):
        print(f"Proxy error details: {e.__class__.__name__}: {str(e)}")


