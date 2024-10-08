import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import httpx
import jieba
import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]
OPENAI_API_BASE = os.path.join(
    os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1/'),
    'chat/completions')

O1_MODEL_LIST = [
    'o1-preview-2024-09-12',
    'o1-mini-2024-09-12',
    'o1-preview',
    'o1-mini',
]


@MODELS.register_module()
class OpenAI(BaseAPIModel):
    """Model wrapper around OpenAI's models.

    Args:
        path (str): The name of OpenAI's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retries if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        openai_api_base (str): The base URL of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        openai_proxy_url (str, optional): An optional proxy URL to use when
            connecting to OpenAI's API. When set to 'ENV', the URL will be
            fetched from the environment variable $OPENAI_PROXY_URL.
            Defaults to None.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): Sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
        tokenizer_path (str, optional): The path to the tokenizer. Use path if
            'tokenizer_path' is None, otherwise use the 'tokenizer_path'.
            Defaults to None.
        extra_body (Dict, optional): Add additional JSON properties to
            the request.
    """

    is_api: bool = True

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = None,
                 openai_api_base: str = OPENAI_API_BASE,
                 openai_proxy_url: Optional[str] = None,
                 mode: str = 'none',
                 logprobs: Optional[bool] = False,
                 top_logprobs: Optional[int] = None,
                 temperature: Optional[float] = None,
                 tokenizer_path: Optional[str] = None,
                 extra_body: Optional[Dict] = None,
                 max_completion_tokens: int = 16384):

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
        self.tokenizer_path = tokenizer_path
        self.hf_tokenizer = None
        self.extra_body = extra_body

        if isinstance(key, str):
            if key == 'ENV':
                if 'OPENAI_API_KEY' not in os.environ:
                    raise ValueError('OpenAI API key is not set.')
                self.keys = os.getenv('OPENAI_API_KEY').split(',')
            else:
                self.keys = [key]
        else:
            self.keys = key

        # Record invalid keys and skip them when requesting API
        self.invalid_keys = set()

        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = openai_api_base

        if openai_proxy_url == 'ENV':
            if 'OPENAI_PROXY_URL' not in os.environ:
                raise ValueError('OPENAI_PROXY_URL is not set.')
            self.proxy_url = os.getenv('OPENAI_PROXY_URL')
        else:
            self.proxy_url = openai_proxy_url

        self.path = path
        self.max_completion_tokens = max_completion_tokens
        self.logger.warning(
            f'Max Completion tokens for {path} is: {max_completion_tokens}')

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 temperature: float = 0.7,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): Sampling temperature to use,
                between 0 and 2. Defaults to 0.7.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.temperature is not None:
            temperature = self.temperature

        def safe_generate(input):
            try:
                return self._generate(input, max_out_len, temperature)
            except Exception as e:
                self.logger.error(f"Error processing input: {str(e)}")
                return "**********Error processing input**********"

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(safe_generate, inputs))
        return results

    def _generate(self, input: PromptType, max_out_len: int,
                  temperature: float) -> str:
        assert isinstance(input, (str, PromptList))

        context_window = 32768
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 8192
        elif 'gpt-3.5' in self.path:
            context_window = 4097

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

        try:
            max_out_len = min(
                max_out_len,
                context_window - self.get_token_len(str(input)) - 100)
        except Exception as e:
            self.logger.error(f"Error in token length calculation: {e}")
            max_out_len = max_out_len
        if max_out_len <= 0:
            return ''

        num_retries = 0
        while num_retries < self.retry:
            self.wait()

            with Lock():
                if len(self.invalid_keys) == len(self.keys):
                    self.logger.error("All API keys are invalid")
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
                if self.path in O1_MODEL_LIST:
                    self.logger.warning(
                        f"'max_token' is unsupported for model {self.path}")
                    self.logger.warning(
                        f'Using max_completion_tokens: {self.max_completion_tokens} for this query')
                    data = dict(
                        model=self.path,
                        messages=messages,
                        max_completion_tokens=self.max_completion_tokens,
                        n=1,
                        logprobs=self.logprobs,
                        top_logprobs=self.top_logprobs,
                        stop=None,
                        temperature=temperature,
                    )
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
                if self.extra_body:
                    data.update(self.extra_body)
                if isinstance(self.url, list):
                    import random
                    url = self.url[random.randint(0, len(self.url) - 1)]
                else:
                    url = self.url

                if self.proxy_url is None:
                    raw_response = requests.post(url,
                                                 headers=header,
                                                 data=json.dumps(data))
                else:
                    proxies = {
                        'http': self.proxy_url,
                        'https': self.proxy_url,
                    }
                    raw_response = requests.post(
                        url,
                        headers=header,
                        data=json.dumps(data),
                        proxies=proxies,
                    )

            except requests.ConnectionError:
                self.logger.error('Connection error, retrying...')
                num_retries += 1
                continue

            try:
                response = raw_response.json()
            except json.JSONDecodeError:
                self.logger.error('JSONDecodeError, got:', raw_response.content)
                num_retries += 1
                continue

            self.logger.debug(str(response))
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
                self.logger.error('Error in response:', response['error'])
                return f"**********API Error: {error.get('message', 'Unknown error')}**********"
            else:
                return "**********Unexpected response format**********"

            num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')

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
        """Get lengths of the tokenized string."""
        assert self.tokenizer_path or self.path
        try:
            tokenizer_path = self.tokenizer_path if self.tokenizer_path else self.path
            try:
                enc = self.tiktoken.encoding_for_model(tokenizer_path)
                return len(enc.encode(prompt))
            except Exception as e:
                self.logger.warning(f'{e}, tiktoken cannot load {tokenizer_path}')
                from transformers import AutoTokenizer
                if self.hf_tokenizer is None:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path, trust_remote_code=True)
                    self.logger.info(f'Tokenizer loaded from {tokenizer_path}')
                return len(self.hf_tokenizer(prompt).input_ids)
        except Exception:
            self.logger.warning('Cannot get tokenizer automatically, using default tokenizer gpt-4.')
            default_tokenizer = 'gpt-4'
            enc = self.tiktoken.encoding_for_model(default_tokenizer)
            return len(enc.encode(prompt))

    def bin_trim(self, prompt: str, num_token: int) -> str:
        """Trim the prompt to a maximum number of tokens."""
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
                 extra_body: Dict | None = None,
                 max_completion_tokens: int = 16384,
                 verbose: bool = False):
        super().__init__(path,
                         max_seq_len,
                         query_per_second,
                         rpm_verbose,
                         retry,
                         key,
                         org,
                         meta_template,
                         openai_api_base,
                         openai_proxy_url,
                         mode,
                         logprobs,
                         top_logprobs,
                         temperature,
                         tokenizer_path,
                         extra_body,
                         verbose=verbose,
                         max_completion_tokens=max_completion_tokens)
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
        if self.verbose:
            self.logger.info(f'Used openai_client: {self.openai_client}')

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

            if self.path in O1_MODEL_LIST:
                self.logger.warning(
                    f"'max_token' is unsupported for model {self.path}")
                self.logger.warning(
                    f'We use max_completion_tokens:'
                    f'{self.max_completion_tokens}for this query')
                query_data = dict(
                    model=self.path,
                    max_completion_tokens=self.max_completion_tokens,
                    n=1,
                    temperature=self.temperature,
                    messages=messages,
                    extra_body=self.extra_body,
                )
            else:
                query_data = dict(
                    model=self.path,
                    max_tokens=max_out_len,
                    n=1,
                    temperature=self.temperature,
                    messages=messages,
                    extra_body=self.extra_body,
                )

            try:
                if self.verbose:
                    self.logger.info('Start calling OpenAI API')
                responses = self.openai_client.chat.completions.create(
                    **query_data)
                if self.verbose:
                    self.logger.info(
                        'Successfully get response from OpenAI API')
                    try:
                        self.logger.info(responses)
                    except Exception as e:  # noqa F841
                        pass
                return responses.choices[0].message.content
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling OpenAI API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')