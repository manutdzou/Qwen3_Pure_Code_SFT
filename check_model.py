from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
#model_name = "Qwen3-4B-250426"
#model_name = "Qwen3-4B-SFT-lite-cold-start"
#model_name = "Qwen3-4B-SFT-heavy-cold-start"
model_name = "Qwen3-4B-QAT-test"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


messages1 = [
    {"role": "user", "content": "以下信息来自你的知识库，可能对回答某些问题有帮助:\n相关资料：个人网络贷款合同（2021 年版） 中国银行股份有限公司个人网络贷款合同 （适用“青春 E 贷·启航贷”业务） 附件 2-1 借款人： 身份证件号码： 联系电话： 贷款人：中国银行股份有限公司 借款人、贷款人根据有关法律、法规，在平等、自愿、诚信的基础上，经协商一致达成如下协议。\\n\\n第一部分：贷款信息 借款金额：小写： 元；大写： 元；贷款币种：人民币； 贷款期限：期限为_____个月，自贷款人实际放款日起算； 贷款利率： %/年（如无特殊说明，本合同项下贷款利率均为采用单利方法计算的年化利率）； 贷款利率定价方式：固定利率； （参考利率：\\n', '，或者借款人还款账户被国家有权机关采取强制措施而导致贷款人无法正常扣收，或者存在其他影响借款人自助贷款操作正常进行的非贷款人过错的 情况，由此产生的一切损失由借款人自行承担。\\n\\n第六条 提前还款 1.除本合同另有约定外，借款人可提前偿还贷款的全部本金和利息，或部分偿还贷款的本金和利息。\\n\\n2.借款人通过贷款人电子服务渠道办理还款业务时，贷款的归还均以贷款人系统记录作为依据，借款 人应及时查询还款交易是否成功。若由于借款人过错（包括但不限于未按相关渠道提示操作或未及时查询 还款交易是否成功等原因）导致贷款未能及时归还，借款人自行承担由此引起的损失。\\n\\n3.借款人根据贷款人电子服务渠道的提示输入提前还款的金额，贷款人扣收借款人提前归还的本息。\\n\\n借款人的自助提前还款账户为本借款合同第五条第 2 款所指定的还款账户或经由借款人确认的其他还款账 户，扣收提前还款金额的方式亦同第五条第 2 款约定。\\n\\n4.借款人实际自助提前还款资金到账时间以贷款人扣款交易时间为准。\\n\\n5.借款人提前部分还款的，贷款人系统会按照“剩余还款期数不变，每期还款金额减少”重新计算借 款人剩余应还款本息，提前还款后每期归还本息以实际扣款为准。\\n\\n\\n\\n', '浮动加点值为 275BP，实际执行年利率 6.6%；③贷款期限为 120 个月，贷款利率为 2021 年 11 月 22 日全国银行间同业拆借中心公布的 5 年期以上贷款 市场报价利率，在参考利率基础上的浮动加点值为 255BP，实际执行年利率 7.2%。） 贷款还款账户：户名 ；还款账号 ； 还款方式： ； 。\\n\\n第二部分：合同通用条款 第一条 贷款金额、币种及期限 1.贷款人同意向借款人提供贷款的金额、币种、期限见第一部分。\\n\\n2.借款人通过贷款人提供的电子服务渠道发起贷款申请，贷款人有权对借款人进行资信调查，并依据 借款人的资信状况决定是否批准借款人贷款申请。\\n\\n\\n\\n\n请记住以上知识库的信息。"},
    {"role": "assistant", "content": "好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？"},
    {"role": "user", "content": "问题：贷款币种是什么？\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造答案; 回答简洁而口语化，语义流畅而连贯。如果找不到答案，请用特殊表情符号😓 拒绝回答。"},
]

messages2 = [
    {"role": "user", "content": "以下信息来自你的知识库，可能对回答某些问题有帮助:\n相关资料：挑战。传统物流行业的操作模式已经不适应现代的物流行业，如何缩短物流过程，降低产品库存，加速对市场的反应，这是所有企业所面临的 问题。吉林省明日科技有限公司开发的《物流管理系统》就是针对这些问题并根据中小型企 业的实际需求而开发的，本系统能够帮助企业实现对物流全过程的优化调度和动态控制，高效整合企业的物流业务，以全面提高经济效益和效率为目的，提供高效、实用、技术的物流 管理系统和运营手段。\\n\\n\\n\\n', '：普通货车（辆）______________ 3 装卸设备（台）______________                            8      专用货车（辆）______________ 4 物流计算机信息管理系统（套）_____________               9      其中：冷藏车（辆）_____________ \\n\\n', ' 5、随时随地通过 IE 浏览器进行数据的统计和查询； 6、减少劳动力，节省成本； 1 颂尼供应链 RFID 数据管理系统软件 V1.0——操作说明书 2 7、客户订制化个性化服务； 1、用户管理界面 可实现在线注册，并可根据客户需求进行权限管理，确保供应链管理系统的数 据安全，提高管理效率。\\n\\n图 1 用户管理界面 2、供应链管理系统主界面 RFID 标签贯穿于包括采购材料，物流,运输，制造，零售、商品库存管理等 全过程。通过 RFID 标签携带的编码可以查询产品在供应链流动的所有细节信息： 商品信息、价格、配送信息、财务信息、订单信息、规划预测、支付、销售等。\\n\\n图 1 为基于 RFID 技术的服装行业供应链管理系统软件主界面,系统主要包括生产 智能化管理系统、仓储智能化管理系统、销售智能化管理系统。\\n\\n\\n\\n\n请记住以上知识库的信息。"},
    {"role": "assistant", "content": "好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？"},
    {"role": "user", "content": "问题：该手册是什么时候发布的？\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造答案; 回答简洁而口语化，语义流畅而连贯。如果找不到答案，请用特殊表情符号😓 拒绝回答。"},
]


messages3 = [{"role": "user", "content": "一个直角三角形，两条边分别为6和8，求第三条边，请列出所有可能？"}]


for mess in [messages1,messages2,messages3]:
    text = tokenizer.apply_chat_template(
        mess,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        repetition_penalty=1.05,
        do_sample=True,
    )

    for i in range(3):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            generation_config=generation_config
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print("\n")
        print(generated_ids)
        print(response)
