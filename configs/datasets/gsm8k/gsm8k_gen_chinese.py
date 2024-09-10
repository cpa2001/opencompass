from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg_5shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="""请一步一步思考，输出最终答案前用“####”标记。下面是五个例子：

问题：Natalia在四月份向她的48个朋友出售了夹子，然后在五月份卖出了四月份的一半。Natalia在四月和五月总共卖了多少个夹子？
答案：Natalia在五月份卖出了48/2 = 24个夹子。
Natalia在四月和五月总共卖出了48+24 = 72个夹子。
#### 72

问题：翁做保姆工作每小时赚12美元。昨天，她只做了50分钟的保姆工作。她赚了多少钱？
答案：翁每分钟赚12/60 = 0.2美元。
工作了50分钟，她赚了0.2 x 50 = 10美元。
#### 10

问题：贝蒂正在为一只价值100美元的新钱包存钱。贝蒂只有她需要的一半的钱。她的父母决定给她15美元，她的祖父母给她的钱是她父母的两倍。贝蒂还需要多少钱才能买到钱包？
答案：起初，贝蒂只有100 / 2 = 50美元。
贝蒂的祖父母给了她15 * 2 = 30美元。
这意味着，贝蒂还需要100 - 50 - 30 - 15 = 5美元。
#### 5

问题：朱莉正在读一本120页的书。昨天，她读了12页，今天她读了昨天的两倍。如果她想明天读剩下的一半页数，她应该读多少页？
答案：今天她读了12 x 2 = 24页。
所以自昨天以来，她一共读了12 + 24 = 36页。
剩下要读的页数是120 - 36 = 84页。
因为她想明天读剩下的一半页数，所以她应该读84/2 = 42页。
#### 42

问题：詹姆斯每周向2个不同的朋友写一封3页的信，每周写两次。他一年写了多少页的信？
答案：他每周给每个朋友写3*2=6页的信
所以他每周写6*2=12页的信
这意味着他每年写12*52=624页的信
#### 624

现在回答下面问题：
问题：{question}
答案：
"""),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsm8k_eval_cfg_5shot = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
                      pred_role='BOT',)

gsm8k_chinese_datasets = [
    dict(
        abbr='gsm8k_chinese_5shot',
        type=GSM8KDataset,
        path='/mnt/petrelfs/chenpengan/opencompass/data/gsm8k_zh',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg_5shot,
        eval_cfg=gsm8k_eval_cfg_5shot),
]