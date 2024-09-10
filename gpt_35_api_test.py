import os
from openai import AzureOpenAI
import httpx
import requests
from urllib.parse import quote
import subprocess

# 设置详细的日志记录
import logging
logging.basicConfig(level=logging.DEBUG)

# 设置代理
proxy_username = "chenpengan"
proxy_password = "IcHMOlFy8Qtu7PRR8oVnkXRhVY6n0fjz85WRocrlvYV1GgF9YZdhgTLd9ADq"
proxy_host = "10.1.20.50"
proxy_port = "23128"

# 对用户名和密码进行URL编码
encoded_username = quote(proxy_username)
encoded_password = quote(proxy_password)

proxy_url = f"http://{encoded_username}:{encoded_password}@{proxy_host}:{proxy_port}"

proxies = {
    "http://": proxy_url,
    "https://": proxy_url
}

# 设置环境变量
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

# 设置更长的超时时间
timeout = httpx.Timeout(30.0, connect=15.0, read=20.0)

def test_proxy():
    try:
        result = subprocess.run(['curl', '-x', proxy_url, 'https://api.ipify.org'], capture_output=True, text=True, timeout=10)
        print(f"Proxy test result: {result.stdout}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Proxy test timed out")
        return False
    except Exception as e:
        print(f"Proxy test failed: {e}")
        return False

def test_connection(url):
    try:
        response = requests.get(url, timeout=15, proxies=proxies)
        print(f"Connection to {url} successful. Status code: {response.status_code}")
        return True
    except requests.Timeout:
        print(f"Connection to {url} timed out")
        return False
    except requests.RequestException as e:
        print(f"Connection to {url} failed: {e}")
        return False

try:
    # 测试代理
    if not test_proxy():
        raise Exception("Proxy connection failed")

    # 测试基本连接
    if not test_connection("https://gpt-35-turbo-instruct-zn.openai.azure.com/"):
        raise Exception("Cannot connect to Azure OpenAI endpoint")

    # 创建 AzureOpenAI 客户端
    client = AzureOpenAI(
        azure_endpoint="https://gpt-35-turbo-instruct-zn.openai.azure.com/",
        api_key="a55c9d72f7ef48d7b637923d6e8815ed",
        api_version="2024-06-01",
        timeout=timeout,
        http_client=httpx.Client(proxies=proxies, timeout=timeout)
    )

    # 尝试 API 调用
    response = client.completions.create(
        model="gpt-35-turbo-instruct",
        prompt="Does Azure OpenAI support customer managed keys?",
        max_tokens=100
    )
    print(response.choices[0].text)
except httpx.TimeoutException as e:
    print(f"Request timed out: {e}")
    print("Consider increasing the timeout or check if the Azure OpenAI service is responding slowly.")
except httpx.ProxyError as e:
    print(f"Proxy error: {e}")
    print("Please check your proxy settings.")
except httpx.ConnectError as e:
    print(f"Connection error: {e}")
    print("Please check your network connection and firewall settings.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print(f"Error type: {type(e)}")
    if hasattr(e, '__dict__'):
        print(f"Error attributes: {e.__dict__}")