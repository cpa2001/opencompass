import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import jieba
import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]
OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'

@MODELS.register_module()
class OpenAI(BaseAPIModel):
    is_api: bool = True

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: float = 0.5,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = None,
                 openai_api_base: str = OPENAI_API_BASE,
                 mode: str = 'none',
                 logprobs: Optional[bool] = False,
                 top_logprobs: Optional[int] = None,
                 temperature: Optional[float] = None):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry)
        import tiktoken
        self.tiktoken = tiktoken
        self.temperature = temperature
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

        if isinstance(key, str):
            if key == 'ENV':
                if 'OPENAI_API_KEY' not in os.environ:
                    raise ValueError('OpenAI API key is not set.')
                self.keys = os.getenv('OPENAI_API_KEY').split(',')
            else:
                self.keys = [key]
        else:
            self.keys = key

        self.invalid_keys = set()
        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = openai_api_base
        self.path = path

    def generate(self, inputs: List[PromptType], max_out_len: int = 512, temperature: float = 0.7, **kwargs) -> List[str]:
        results = []
        for input in inputs:
            try:
                result = self._generate(input, max_out_len, temperature)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing input: {str(e)}")
                results.append("**********Error processing input**********")
        return results

    def _generate(self, input: PromptType, max_out_len: int, temperature: float) -> str:
        assert isinstance(input, (str, PromptList))

        context_window = 4096
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 8192

        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        max_out_len = min(max_out_len, context_window - self.get_token_len(str(input)) - 100)
        if max_out_len <= 0:
            return ''

        self.wait()

        with Lock():
            if len(self.invalid_keys) == len(self.keys):
                return "**********All API keys are invalid**********"

            while True:
                self.key_ctr += 1
                if self.key_ctr == len(self.keys):
                    self.key_ctr = 0
                if self.keys[self.key_ctr] not in self.invalid_keys:
                    break

            key = self.keys[self.key_ctr]

        header = {
            'Authorization': f'Bearer {key}',
            'content-type': 'application/json',
            'api-key': key,
        }

        if self.orgs:
            with Lock():
                self.org_ctr += 1
                if self.org_ctr == len(self.orgs):
                    self.org_ctr = 0
            header['OpenAI-Organization'] = self.orgs[self.org_ctr]

        try:
            if 'gpt-3.5-turbo-instruct' in self.path:
                prompt = "\n".join([msg['content'] for msg in messages])
                data = {
                    'model': self.path,
                    'prompt': prompt,
                    'max_tokens': max_out_len,
                    'temperature': temperature
                }
            else:
                data = dict(
                    model=self.path,
                    messages=messages,
                    max_tokens=max_out_len,
                    n=1,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    stop=None,
                    temperature=temperature,
                )
            raw_response = requests.post(self.url, headers=header, data=json.dumps(data))
            raw_response.raise_for_status()
            response = raw_response.json()
        except requests.RequestException as e:
            return f"**********API Request Error: {str(e)}**********"

        if 'choices' in response:
            choice = response['choices'][0]
            if choice.get('finish_reason') == 'content_filter':
                return "**********Content filtered**********"
            content = self._extract_content(choice)
            if content:
                return content.strip()
            else:
                return "**********Unexpected response structure**********"
        elif 'error' in response:
            error = response['error']
            if error.get('code') == 'content_filter':
                return "**********Content filtered**********"
            return f"**********API Error: {error.get('message', 'Unknown error')}**********"
        else:
            return "**********Unexpected response format**********"

    def _extract_content(self, choice):
        if 'gpt-3.5-turbo-instruct' in self.path:
            return choice.get('text', '')
        elif 'message' in choice and 'content' in choice['message']:
            return choice['message']['content']
        elif 'text' in choice:
            return choice['text']
        elif 'content' in choice:
            return choice['content']
        return None

    def get_token_len(self, prompt: str) -> int:
        enc = self.tiktoken.encoding_for_model(self.path)
        return len(enc.encode(prompt))

    def bin_trim(self, prompt: str, num_token: int) -> str:
        token_len = self.get_token_len(prompt)
        if token_len <= num_token:
            return prompt
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if pattern.search(prompt):
            words = list(jieba.cut(prompt, cut_all=False))
            sep = ''
        else:
            words = prompt.split(' ')
            sep = ' '

        l, r = 1, len(words)
        while l + 2 < r:
            mid = (l + r) // 2
            if self.mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif self.mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
            elif self.mode == 'rear':
                cur_prompt = sep.join(words[:mid])

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid
            else:
                r = mid

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt

class OpenAISDK(OpenAI):

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: str | List[str] = 'ENV',
                 org: str | List[str] | None = None,
                 meta_template: Dict | None = None,
                 openai_api_base: str = OPENAI_API_BASE,
                 openai_proxy_url: Optional[str] = None,
                 mode: str = 'none',
                 logprobs: bool | None = False,
                 top_logprobs: int | None = None,
                 temperature: float | None = None,
                 tokenizer_path: str | None = None,
                 extra_body: Dict | None = None):
        super().__init__(path, max_seq_len, query_per_second, rpm_verbose,
                         retry, key, org, meta_template, openai_api_base,
                         openai_proxy_url, mode, logprobs, top_logprobs,
                         temperature, tokenizer_path, extra_body)
        from openai import OpenAI

        if self.proxy_url is None:
            self.openai_client = OpenAI(base_url=openai_api_base, api_key=key)
        else:
            proxies = {
                'http://': self.proxy_url,
                'https://': self.proxy_url,
            }

            self.openai_client = OpenAI(
                base_url=openai_api_base,
                api_key=key,
                http_client=httpx.Client(proxies=proxies))

    def _generate(self, input: PromptList | str, max_out_len: int,
                  temperature: float) -> str:
        assert isinstance(input, (str, PromptList))

        # max num token for gpt-3.5-turbo is 4097
        # Most models' token limits are above 32k
        context_window = 32768
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 8192
        elif 'gpt-3.5' in self.path:
            context_window = 4097

        # will leave 100 tokens as prompt buffer, triggered if input is str
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        # Hold out 100 tokens due to potential errors in tiktoken calculation
        # try:
        #     max_out_len = min(
        #         max_out_len,
        #         context_window - self.get_token_len(str(input)) - 100)
        # except KeyError:
        #     max_out_len = max_out_len
        # if max_out_len <= 0:
        #     return ''

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                responses = self.openai_client.chat.completions.create(
                    model=self.path,
                    max_tokens=max_out_len,
                    n=1,
                    temperature=self.temperature,
                    messages=messages,
                    extra_body=self.extra_body,
                )
                return responses.choices[0].message.content
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling OpenAI API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')