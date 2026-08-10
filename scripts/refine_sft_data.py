"""
SFT 数据精炼管道

对 extract_sft_from_pdf.py 产出的原始数据进行全面质量处理：
  1. 噪音清洗：英文混入、集数标记、非教学内容
  2. 上下文断裂修复：代词/连接词开头
  3. 长度控制：<200 过滤、>800 切分
  4. 指令精化：消除通用模板、差异化生成
  5. 去重：output 内容去重
  6. 数据集划分：90/10 train/val
  7. 纯 PDF 口语化数据（不混入 formulas_db 正式文本，避免稀释口语信号）

输入: data/processed/sft_train_pdf.jsonl (原始 PDF 提取数据)
输出: data/processed/sft_train_final.jsonl  (最终训练集)
      data/processed/sft_val_final.jsonl    (验证集)
      data/processed/sft_refine_report.json (质量报告)
"""

import json
import re
import os
import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PDF = PROJECT_ROOT / "data" / "processed" / "sft_train_pdf.jsonl"
INPUT_DB = PROJECT_ROOT / "data" / "processed" / "sft_train_p1.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================
# 1. 噪音清洗
# ============================================================

# 医学/ regulatory 缩写（保留，有教学价值）
KEEP_ABBREVIATIONS = {
    'EKG', 'EEG', 'FDA', 'AIDS', 'HIV', 'SARS', 'ALS', 'GMP', 'DNA', 'RNA',
    'CT', 'MRI', 'ICU', 'CPR', 'BMI', 'BP', 'HBV', 'HCV',
}

# 非教学性英文（需删除）——删除所有 3+ 字母英文词，除了医学缩写
def clean_english(text: str) -> str:
    """删除所有非缩写英文单词"""
    def replace_english(match):
        word = match.group(0)
        if word.upper() in KEEP_ABBREVIATIONS:
            return word
        return ''
    text = re.sub(r'[a-zA-Z]{2,}', replace_english, text)
    # 清理孤立的英文字母
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    # 清理因删除英文留下的多余空格和标点
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 清理行尾空格
    text = '\n'.join(l.rstrip() for l in text.split('\n'))
    return text

# 集数标记
EPISODE_MARKER = re.compile(r'\[第\d+\s*集\]')

# 非教学内容判定关键词
NON_TEACHING_MARKERS = [
    "倪海厦先生及其", "1954 年生于", "倪海厦(the last hope",
    "美国加州中医药大学博士指导教授", "美国汉唐中医",
    "目录", "········", "视频目录", "课程目录",
    "易筋经", "动作示范", "我演示给",
    "版权所有", "购买完整版", "扫码获取",
]

# 目录页判定：大量短行 + 点号连接
def is_toc_page(text: str) -> bool:
    """检测是否为目录页"""
    # 中文点号
    if text.count("·····") > 3:
        return True
    # ASCII 点号连续（如 "刺齐论........... 251"）
    if text.count("...") > 3:
        return True
    # 点号+页码模式（如 "狼牙汤..............304"）
    if re.search(r'\.{3,}\s*\d+', text):
        return True
    # 编号+点号模式（如 "二六一、白垩........ 241"）
    if re.search(r'[一二三四五六七八九十百千\d]+[、.]\s*[\u4e00-\u9fff]+\s*\.{3,}', text):
        return True
    lines = text.split("\n")
    if len(lines) > 5:
        short_lines = sum(1 for l in lines if len(l.strip()) < 15)
        if short_lines > len(lines) * 0.7:
            return True
    return False

def is_biography(text: str) -> bool:
    """检测是否为倪海厦个人生平介绍"""
    bio_markers = ["倪海厦先生及其", "1954 年生于", "倪海厦(the last hope",
                   "美国加州中医药大学博士指导教授", "祖籍浙江瑞安",
                   "汉唐中医学院院长", "佛州针灸委员会委员"]
    hits = sum(1 for m in bio_markers if m in text)
    return hits >= 2

def is_non_teaching(text: str) -> bool:
    """检测是否为非教学内容"""
    if is_biography(text):
        return True
    if is_toc_page(text):
        return True
    for marker in NON_TEACHING_MARKERS:
        if marker in text:
            # 检查是否只是顺带提到（比如在讲解中提到"目录"这个词不算）
            # 只有当 marker 出现在前 200 字内才算非教学
            if text.find(marker) < 200:
                return True
    return False


def clean_noise(text: str) -> str:
    """清洗噪音：英文、集数标记"""
    # 删除集数标记
    text = EPISODE_MARKER.sub("", text)
    # 删除非教学性英文（保留医学缩写）
    text = clean_english(text)
    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 清理行首行尾空格
    text = '\n'.join(l.strip() for l in text.split('\n'))
    return text.strip()


# ============================================================
# 2. 上下文断裂修复
# ============================================================

# 代词/连接词开头（断裂信号）
PRONOUN_STARTS = [
    "所以", "所以说", "因为", "但是", "而且", "因此", "由此",
    "这个", "这个方子", "这个条辨", "这个观念", "这个意思",
    "这个道理", "那这个", "那我们", "那你们", "那他",
    "实际上", "简单讲", "就是这样", "换句话说",
    "你看", "你想想", "你错了", "你想想看",
    "好，", "对，", "不对", "嗯，",
    "现在我们", "现在讲", "现在这个",
    "再来", "接下来", "下面",
]

def fix_context_break(text: str) -> str | None:
    """
    修复上下文断裂。
    策略：如果是代词/连接词开头，去掉开头词，检查剩余内容是否自足。
    如果去掉后仍 < 200 字，返回 None（过滤）。
    """
    for pronoun in sorted(PRONOUN_STARTS, key=len, reverse=True):
        if text.startswith(pronoun):
            remaining = text[len(pronoun):].strip()
            # 去掉开头标点
            remaining = re.sub(r'^[，。、！？：；\s]+', '', remaining)
            if len(remaining) >= 200:
                return remaining
            else:
                return None
    return text


# ============================================================
# 3. 长度控制
# ============================================================

def control_length(text: str, min_len: int = 200, max_len: int = 800) -> list[str]:
    """
    长度控制：< min_len 过滤，> max_len 按句子边界切分。
    返回列表：0 个（过滤）、1 个（合格）、多个（切分后）。
    """
    text = text.strip()
    if len(text) < min_len:
        return []
    if len(text) <= max_len:
        return [text]
    
    # 超长文本按句子切分
    sentences = re.split(r'(?<=[。！？\n])', text)
    chunks = []
    current = ""
    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) > max_len and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += sent
    if current.strip():
        chunks.append(current.strip())
    
    # 过滤切分后太短的块
    return [c for c in chunks if len(c) >= min_len]


# ============================================================
# 4. 指令精化（核心）
# ============================================================

# 条文编号
CLAUSE_PATTERN = re.compile(r'^([一二三四五六七八九十百千]+)[：:]\s*(.+)')

# 扩展方剂库
FORMULAS = [
    "桂枝汤", "麻黄汤", "葛根汤", "小柴胡汤", "大柴胡汤",
    "白虎汤", "白虎加人参汤", "承气汤", "调胃承气汤", "小承气汤", "大承气汤",
    "理中丸", "理中汤", "四逆汤", "真武汤", "附子汤", "桃花汤",
    "黄连阿胶汤", "猪苓汤", "五苓散", "茵陈蒿汤",
    "麻杏石甘汤", "葛根黄芩黄连汤", "黄芩汤",
    "桂枝加葛根汤", "桂枝加附子汤", "桂枝麻黄各半汤",
    "小建中汤", "炙甘草汤", "茯苓四逆汤",
    "乌梅丸", "麻黄升麻汤", "干姜附子汤", "茯苓桂枝白术甘草汤",
    "桂枝去桂加茯苓白术汤", "厚朴生姜半夏甘草人参汤",
    "大青龙汤", "小青龙汤", "桂枝二麻黄一汤",
    "文蛤散", "三物白散", "十枣汤", "瓜蒂散",
    "旋覆代赭汤", "桂枝人参汤", "白虎加桂枝汤",
    "新加汤", "葛根加半夏汤", "桂枝加厚朴杏子汤",
    "白通汤", "白通加猪胆汁汤", "通脉四逆汤", "吴茱萸汤",
    "当归四逆汤", "当归四逆加吴茱萸生姜汤", "麻黄连翘赤小豆汤",
    "栀子豉汤", "栀子生姜豉汤", "栀子厚朴汤", "栀子干姜汤",
    "柴胡加芒硝汤", "柴胡桂枝汤", "柴胡桂枝干姜汤",
    "半夏泻心汤", "生姜泻心汤", "甘草泻心汤",
    "赤石脂禹余粮汤", "抵当汤", "抵当丸", "桃核承气汤",
]

# 证型/病名（精确匹配——排除过于宽泛的词）
SYNDROMES = [
    "太阳病", "阳明病", "少阳病", "太阴病", "少阴病", "厥阴病",
    "太阳中风", "太阳伤寒", "太阳温病", "风温",
    "结胸", "痞证", "蓄水证", "蓄血证",
    "脏结", "热入血室", "合病", "并病", "两感",
    "直中", "越经传",
    "表虚证", "表实证", "半表半里证",
    "阳明腑实", "阳明经证", "少阳证",
    "少阴寒化", "少阴热化", "厥阴厥逆",
    "亡阳证", "亡阴证",
]
# 注意："伤寒"、"中风"、"温病" 单独出现时过于宽泛，不做通用匹配

# 药材名
HERBS = [
    "桂枝", "麻黄", "芍药", "白芍", "甘草", "生姜", "大枣",
    "葛根", "柴胡", "黄芩", "半夏", "人参", "干姜", "炮附子",
    "附子", "茯苓", "白术", "黄连", "大黄", "芒硝",
    "杏仁", "石膏", "知母", "粳米", "厚朴", "枳实",
    "当归", "川芎", "生地黄", "熟地黄", "阿胶", "鸡子黄",
    "猪苓", "泽泻", "滑石", "茵陈", "栀子", "黄柏",
    "细辛", "五味子", "防风", "羌活", "独活",
    "麦冬", "天冬", "百合", "旋覆花", "代赭石",
    "桃仁", "水蛭", "虻虫", "牡丹皮", "赤芍",
    "瓜蒌", "薤白", "半夏", "乌梅", "花椒",
    "吴茱萸", "黄连", "秦皮", "白头翁",
    "赤石脂", "禹余粮", "文蛤", "贝母",
    # 峻下逐水药
    "甘遂", "大戟", "芫花", "商陆", "牵牛子", "巴豆",
    # 化痰止咳药
    "葶苈子", "白前", "前胡", "竹茹", "竹沥", "天竺黄",
    "海藻", "昆布", "海浮石", "礞石", "瓦楞子",
    # 理气药
    "木香", "香附", "乌药", "沉香", "檀香", "藿香",
    "佩兰", "砂仁", "白豆蔻", "草豆蔻", "草果",
    # 活血化瘀药
    "乳香", "没药", "延胡索", "郁金", "姜黄", "莪术",
    "三棱", "益母草", "泽兰", "牛膝", "王不留行",
    # 补虚药
    "黄芪", "白扁豆", "山药", "太子参", "西洋参",
    "肉苁蓉", "锁阳", "巴戟天", "补骨脂", "益智仁",
    "菟丝子", "沙苑子", "蛤蚧", "冬虫夏草", "紫河车",
    # 安神药
    "朱砂", "磁石", "龙骨", "琥珀", "酸枣仁", "柏子仁", "远志",
    # 平肝息风药
    "羚羊角", "天麻", "钩藤", "地龙", "全蝎", "蜈蚣",
    "僵蚕", "珍珠母", "牡蛎", "珍珠",
    # 其他
    "灶心土", "乱发", "人尿", "猪胆汁", "猪肤", "猪膏",
    "蜀漆", "常山", "白蔹", "白及", "连翘", "紫参",
    "蜣螂", "鼠妇", "蜂窠", "蟅虫", "蛴螬",
    "石灰", "铅丹", "雄黄", "矾石", "硝石",
]

# 针灸特定穴/经络
ACU_TOPICS = [
    "任脉", "督脉", "冲脉", "带脉",
    "手太阴肺经", "手阳明大肠经", "足阳明胃经", "足太阴脾经",
    "手少阴心经", "手太阳小肠经", "足太阳膀胱经", "足少阴肾经",
    "手厥阴心包经", "手少阳三焦经", "足少阳胆经", "足厥阴肝经",
    # 经络缩写（倪师讲课中常用）
    "肺经", "大肠经", "胃经", "脾经", "心经", "小肠经",
    "膀胱经", "肾经", "心包经", "三焦经", "胆经", "肝经",
    "奇经八脉", "十五络脉", "十二经脉",
    "井穴", "荥穴", "俞穴", "经穴", "合穴", "原穴", "络穴",
    "郄穴", "募穴", "背俞穴", "八会穴", "下合穴",
    "五输穴", "五俞穴", "俞募治疗", "原络治疗", "会郄治疗",
    # 常用穴位
    "合谷", "曲池", "足三里", "三阴交", "太冲", "太溪",
    "神门", "内关", "百会", "关元", "气海", "中脘",
    "膻中", "命门", "肾俞", "肝俞", "脾俞", "心俞", "肺俞",
    "太白", "商丘", "内庭", "阴陵泉", "地机", "血海",
    "列缺", "尺泽", "鱼际", "少商",
    "迎香", "禾髎",
    "承泣", "四白", "颊车", "下关", "头维",
    "大横", "腹哀",
    "极泉", "少海", "神门", "少府",
    "少泽", "后溪", "养老", "小海",
    "睛明", "攒竹", "天柱", "委中", "承山", "昆仑", "申脉",
    "涌泉", "然谷", "太溪", "复溜", "照海",
    "天池", "曲泽", "间使", "大陵", "劳宫",
    "关冲", "中渚", "外关", "支沟", "天井",
    "瞳子髎", "听会", "率谷", "阳白", "风池", "肩井", "环跳", "风市", "阳陵泉", "悬钟",
    "大敦", "行间", "太冲", "中封", "曲泉",
    "针刺法", "补泻", "得气", "留针", "迎随补泻",
    "艾灸", "温针灸", "直接灸", "间接灸",
    "梅花针", "三棱针", "九针",
]

# 内经理论
NEIJING_TOPICS = [
    "阴阳", "五行", "木火土金水", "相生", "相克", "相乘", "相侮",
    "脏腑", "心", "肝", "脾", "肺", "肾", "心包",
    "胆", "胃", "小肠", "大肠", "膀胱", "三焦",
    "气血", "津液", "营卫", "精气", "神",
    "望诊", "闻诊", "问诊", "切诊", "脉诊", "舌诊",
    "治病求本", "正治", "反治", "标本", "扶正", "祛邪",
    "四气调神", "生气通天", "上古天真",
]

# 本草理论
BENCAO_TOPICS = [
    "四气", "五味", "寒热温凉", "辛甘酸苦咸",
    "归经", "升降浮沉", "有毒", "无毒",
    "炮制", "修治", "水制", "火制", "炙", "炒", "煅", "煨", "蒸",
    "君臣佐使", "七情", "相须", "相使", "相畏", "相杀", "相恶", "相反",
    "十八反", "十九畏", "用药禁忌",
    "解表药", "清热药", "泻下药", "祛风湿药", "化湿药",
    "利水渗湿药", "温里药", "理气药", "消食药", "驱虫药",
    "止血药", "活血化瘀药", "化痰止咳平喘药",
    "安神药", "平肝息风药", "开窍药", "补虚药", "收涩药",
]

# 金匮杂病
JINKUI_TOPICS = [
    "痉病", "湿病", "暍病", "百合病", "狐惑病", "阴阳毒",
    "疟病", "中风", "历节", "血痹", "虚劳",
    "肺痿", "肺痈", "咳嗽", "上气", "奔豚",
    "胸痹", "心痛", "腹满", "寒疝", "宿食",
    "痰饮", "悬饮", "溢饮", "支饮",
    "水气病", "黄疸", "惊悸", "吐血", "下血",
    "呕吐", "哕", "下利", "疮痈", "肠痈",
    "浸淫疮", "趺蹶", "手指臂肿", "转筋", "阴狐疝", "蛔虫",
]

# 伤寒论专题
SHANGHAN_TOPICS = [
    "六经辨证", "传变", "传经", "直中", "合病", "并病",
    "表里", "寒热", "虚实", "阴阳",
    "汗法", "吐法", "下法", "和法", "温法", "清法", "补法", "消法",
    "坏病", "变证", "误治", "过汗",
    "发汗", "解表", "和解", "攻下", "温里", "清热",
]

# 症状/体征（用于主题检测和指令生成）
SYMPTOMS = [
    "恶寒", "恶风", "发热", "潮热", "往来寒热", "但热不寒",
    "头痛", "身痛", "骨节疼痛", "项背强几几", "身重",
    "汗出", "无汗", "盗汗", "自汗", "绝汗",
    "口渴", "咽干", "咽痛", "口苦", "口燥",
    "呕吐", "干呕", "哕逆", "食入即吐",
    "下利", "便秘", "便脓血", "自利",
    "心悸", "烦躁", "谵语", "郑声", "发狂",
    "咳嗽", "喘", "气上冲", "短气", "不得息",
    "小便不利", "小便频数", "遗尿", "小便色白",
    "腹满", "腹痛", "胁痛", "心下痞", "心下悸",
    "心下硬", "按之心下满痛", "腹中急痛",
    "但欲寐", "不得卧", "不得眠", "嗜卧",
    "四肢厥逆", "手足温", "足手寒", "四末冷",
    "衄血", "吐血", "便血", "尿血", "发黄",
    "渴欲饮水", "水入则吐", "消渴",
    "振寒", "战栗", "瘈疭", "拘急",
    "面目浮肿", "一身悉肿", "肤黄",
]

# 治疗/治法概念
TREATMENT_CONCEPTS = [
    "同病异治", "异病同治", "三因制宜",
    "因时制宜", "因地制宜", "因人制宜",
    "扶正祛邪", "标本兼治", "治病求本",
    "正治", "反治", "寒者热之", "热者寒之",
    "虚则补之", "实则泻之", "通因通用", "塞因塞用",
    "上病下治", "下病上治", "从外治内", "从内治外",
    "逆流挽舟", "提壶揭盖", "釜底抽薪",
    "壮水之主", "益火之源", "阳中求阴", "阴中求阳",
]

# 诊断方法概念
DIAGNOSIS_CONCEPTS = [
    "八纲辨证", "六经辨证", "脏腑辨证", "卫气营血辨证", "三焦辨证",
    "病因辨证", "经络辨证",
    "腹诊", "按诊", "触诊",
    "舌质", "舌苔", "舌体", "舌态",
    "浮脉", "沉脉", "迟脉", "数脉", "滑脉", "涩脉",
    "弦脉", "紧脉", "洪脉", "微脉", "芤脉", "革脉",
    "牢脉", "濡脉", "弱脉", "散脉", "细脉", "伏脉",
    "促脉", "结脉", "代脉", "动脉",
]

# 方剂讨论的方面（用于差异化 instruction）
ASPECT_PATTERNS = {
    "组成": ["组成", "由.*组成", "药物", "配方", "里面有", "包含"],
    "主治": ["主治", "治什么", "用于", "适应", "适用", "治疗"],
    "病机": ["病机", "原理", "为什么", "什么意思", "道理", "机制", "机理"],
    "加减": ["加减", "变化", "加减方", "变方", "加减法"],
    "禁忌": ["禁忌", "不宜", "不可", "慎用", "误用"],
    "辨证": ["辨证", "辨证要点", "脉证", "证候", "指征"],
    "鉴别": ["鉴别", "区别", "区分", "比较", "不同", "差异", "对比"],
    "剂量": ["剂量", "用量", "服法", "怎么服", "煎法", "煮法"],
    "配伍": ["配伍", "搭配", "合用", "为什么.*用.*药", "君药", "臣药"],
}

# 证型讨论的方面
SYNDROME_ASPECTS = {
    "病机": ["病机", "原因", "为什么", "什么意思", "机理", "机制"],
    "症状": ["症状", "表现", "临床", "证候", "主要表现"],
    "治疗": ["治疗", "治法", "方剂", "用什么方", "处方"],
    "传变": ["传变", "转归", "发展", "演变", "预后"],
    "辨证": ["辨证", "诊断", "鉴别", "区分", "怎么辨"],
}

# 穴位/经络讨论的方面
ACU_ASPECTS = {
    "定位": ["定位", "位置", "在哪里", "取穴", "找穴"],
    "循行": ["循行", "走向", "路线", "起于", "络", "属"],
    "主治": ["主治", "治什么", "治疗", "功效", "作用"],
    "针刺": ["针刺", "手法", "进针", "深度", "角度", "补泻"],
    "配穴": ["配穴", "配合", "组合", "搭配", "处方"],
}

def get_first_sentences(text: str, n: int = 2) -> str:
    """提取前 n 个句子（用于主题检测，避免全文关键词干扰）"""
    first_line = text.split('\n')[0].strip()
    # 去掉条文编号前缀
    first_line = re.sub(r'^[一二三四五六七八九十百千]+[：:]\s*', '', first_line)
    # 取前 2 个句子
    sentences = re.split(r'(?<=[。！？])', first_line)
    return ''.join(sentences[:n])[:300]


def detect_aspect(text: str, aspect_dict: dict) -> str:
    """检测文本讨论的是哪个方面（只看前 300 字）"""
    head = text[:300]
    for aspect, patterns in aspect_dict.items():
        for p in patterns:
            if re.search(p, head):
                return aspect
    return "general"

def find_topic_in_text(text: str, topics: list[str], search_range: int = 400) -> str | None:
    """在文本前 N 字中查找第一个匹配的主题词"""
    head = text[:search_range]
    for topic in topics:
        if topic in head:
            return topic
    return None

def find_topic_in_first_sentence(text: str, topics: list[str]) -> str | None:
    """只在前 1-2 句中查找主题词（更精准，避免全文关键词干扰）"""
    first_sentences = get_first_sentences(text, n=2)
    for topic in topics:
        if topic in first_sentences:
            return topic
    return None


# ============================================================
# 关键短语质量验证
# ============================================================

# 会话性片段/口语词——出现在关键短语中说明提取质量差
STOP_PHRASES = {
    # 人称代词
    "你", "我", "他", "她", "它", "你们", "我们", "他们",
    # 亲属/人称
    "女儿", "儿子", "先生", "太太", "妈妈", "爸爸", "老师",
    "病人", "患者", "小孩子", "小孩子家",
    # 口语连接词
    "所以", "因为", "但是", "而且", "因此", "然后",
    "实际上", "简单讲", "就是说", "换句话说",
    "同样的", "反正", "总之", "不过", "而且",
    # 口语问答/感叹
    "怎么开", "怎么做", "怎么用", "怎么办", "怎么回事",
    "好不好", "对不对", "是不是", "行不行",
    "为什么", "怎么办呢", "怎么样子",
    "什么样", "问题", "碰到", "遇到", "就碰到",
    # 动作/感叹
    "咬", "打", "骂", "叫", "说", "想", "看",
    "小心", "注意", "记得", "知道", "了解",
    "好", "对", "是", "嗯", "哦", "啊",
    # 非医学概念
    "石灰用", "很毒", "很凶", "很厉害",
    "将来", "以后", "有时候", "有时候要",
}

# 医学相关字符集——关键短语至少包含一个才算有效
MEDICAL_CHARS = set(
    "汤丸散膏丹煎饮片药证病脉症经络穴俞募络针灸"
    "寒热虚实表里阴阳五行脏腑气血营卫"
    "风寒暑湿燥火痰饮瘀血逆陷"
    "浮沉迟数滑涩弦紧洪微芤革牢"
    "促结代动濡弱散细伏"
    "太阳阳明少阳太阴少阴厥阴"
    "汗吐下和温清补消"
    "痉湿暍百合狐惑疟历节血痹虚劳"
    "肺痿肺痈咳嗽上气奔豚胸痹心痛"
    "腹满寒疝宿食痰饮悬饮溢饮支饮"
    "水气黄疸惊悸吐血下利疮痈肠痈"
    "归经炮制君臣佐使七情十八反十九畏"
    "诊脉舌望闻问切"
    "十二任督冲带奇经络脉"
    "解表清热泻下祛风湿化湿利水"
    "温里理气消食驱虫止血活血"
    "化痰止咳平喘安神平肝息风开窍补虚收涩"
)


def validate_key_phrase(phrase: str) -> bool:
    """
    验证关键短语是否达到质量标准。
    返回 True 表示通过验证（可用于生成指令）。
    """
    # 1. 长度检查：3-15 字
    if len(phrase) < 3 or len(phrase) > 15:
        return False
    
    # 2. 不能以停止词开头
    for stop in STOP_PHRASES:
        if phrase.startswith(stop):
            return False
    
    # 2.5. 不能以常见助词/介词/连词开头（v3.3 新增）
    PARTICLE_STARTS = {'了', '到', '去', '在', '就', '那', '这', '如果',
                       '当', '地', '条', '（', '(', '"', '\u201c', '\u2018'}
    if phrase[0] in PARTICLE_STARTS:
        return False
    
    # 3. 不能包含停止词在任何位置（v3.2 加强）
    for stop in STOP_PHRASES:
        if len(stop) >= 2 and stop in phrase:
            return False
    
    # 4. 不能包含明显的人称代词
    if any(w in phrase for w in ["你", "我", "他", "她", "咱们", "大家"]):
        return False
    
    # 5. 不能全是标点或空格
    if not re.search(r'[\u4e00-\u9fff]', phrase):
        return False
    
    # 6. 不能包含句末标点（提取的应该是短语不是句子）
    if any(p in phrase for p in "。！？；"):
        return False
    
    # 7. 不能包含引号或括号（不完整的引用片段）
    if any(p in phrase for p in '""''「」『』""\'\'()[]'):
        return False
    
    # 8. 至少包含一个医学相关字符
    medical_count = sum(1 for c in phrase if c in MEDICAL_CHARS)
    if medical_count < 1:
        return False
    
    # 9. 长短语（>6字）需要更多医学内容（至少2个医学字符）
    if len(phrase) > 6 and medical_count < 2:
        return False
    
    # 10. 不能是纯数字或编号
    if re.match(r'^[一二三四五六七八九十百千万0-9]+$', phrase):
        return False
    
    # 11. 不能包含逗号分隔的多个从句（说明提取了完整句子而非短语）
    if phrase.count('，') > 1:
        return False
    
    # 12. v3.2: 短语必须包含至少一个已识别的医学词（2+字）
    # 这比仅检查单个医学字符更严格、更准确
    if not _contains_medical_term(phrase):
        return False
    
    return True


# 预构建医学词集合（用于快速查找）
_ALL_MEDICAL_TERMS = None

def _get_all_medical_terms() -> set:
    """获取所有医学词的集合（延迟初始化）"""
    global _ALL_MEDICAL_TERMS
    if _ALL_MEDICAL_TERMS is None:
        terms = set()
        for lst in [FORMULAS, SYNDROMES, HERBS, SYMPTOMS, ACU_TOPICS,
                    NEIJING_TOPICS, BENCAO_TOPICS, JINKUI_TOPICS, SHANGHAN_TOPICS,
                    TREATMENT_CONCEPTS, DIAGNOSIS_CONCEPTS]:
            for t in lst:
                if len(t) >= 2:
                    terms.add(t)
        # 额外添加常见医学词
        terms.update([
            "发汗", "解表", "攻下", "温里", "清热", "和解",
            "处方", "辨证", "辨治", "论治", "证候", "病机",
            "组方", "配伍", "剂量", "煎服", "炮制",
            "脾脏", "肝脏", "心脏", "肺脏", "肾脏",
            "脾经", "心经", "肝经", "肺经", "肾经",
            "内出血", "外感", "内伤", "杂病",
        ])
        _ALL_MEDICAL_TERMS = terms
    return _ALL_MEDICAL_TERMS

def _contains_medical_term(phrase: str) -> bool:
    """检查短语是否包含至少一个已识别的医学词"""
    terms = _get_all_medical_terms()
    for term in terms:
        if term in phrase:
            return True
    return False


def clean_key_phrase(phrase: str) -> str:
    """清理关键短语：去除首尾标点、内嵌引号、多余空格"""
    # 去除所有引号和括号（内嵌的也要清理）
    for q in '\u201c\u201d\u2018\u2019「」『』""\'\'()[]':
        phrase = phrase.replace(q, '')
    # 去除首尾标点
    strip_chars = '，。、！？：； \t\n'
    phrase = phrase.strip(strip_chars)
    # 去除中间多余的逗号（保留一个）
    phrase = re.sub(r'，{2,}', '，', phrase)
    # 限制长度
    if len(phrase) > 15:
        # 在标点处截断
        for i, c in enumerate(phrase):
            if c in "，、：；" and i >= 3:
                phrase = phrase[:i]
                break
        phrase = phrase[:15]
    return phrase.strip()


# ============================================================
# 全文医学术语扫描
# ============================================================

# 合并所有主题列表用于全文扫描（按优先级排序）
ALL_MEDICAL_TOPICS = (
    [(t, "formula") for t in FORMULAS]
    + [(t, "syndrome") for t in SYNDROMES]
    + [(t, "herb") for t in HERBS]
    + [(t, "symptom") for t in SYMPTOMS]
    + [(t, "acupuncture") for t in ACU_TOPICS]
    + [(t, "neijing") for t in NEIJING_TOPICS if len(t) >= 2]
    + [(t, "bencao") for t in BENCAO_TOPICS]
    + [(t, "jinkui") for t in JINKUI_TOPICS]
    + [(t, "shanghan") for t in SHANGHAN_TOPICS]
    + [(t, "treatment") for t in TREATMENT_CONCEPTS]
    + [(t, "diagnosis") for t in DIAGNOSIS_CONCEPTS]
    + [(t, "concept") for t in [
        "营卫不和", "传经", "六经", "辨证", "经方", "时方",
        "表里", "寒热", "虚实", "脉象",
        "恶寒", "恶风", "发热", "汗出", "无汗", "头痛",
        "体痛", "呕逆", "下利", "烦躁", "口渴", "心下痞",
        "但热不寒", "往来寒热", "四肢厥逆",
    ]]
)


def scan_text_for_medical_term(text: str, book: str = "") -> tuple[str, str] | None:
    """
    全文扫描医学术语，按特异性和词频选择最佳匹配。
    
    v3 改进策略：
    1. 收集全文中所有匹配的医学词
    2. 按长度（特异性）降序排列——更长的词更具体
    3. 同长度时按词频降序——出现次数多的更可能是主题
    4. 过于通用的词（阴阳/五行等）降低优先级
    
    返回 (term, category) 或 None。
    """
    all_matches = []  # [(term, category, freq, length), ...]
    
    # 过于通用的术语——除非没有其他匹配，否则不用
    VERY_COMMON = {"阴阳", "五行", "气血", "津液", "营卫", "精气",
                   "表里", "寒热", "虚实", "心", "肝", "脾", "肺", "肾",
                   "神", "胆", "胃", "小肠", "大肠", "膀胱", "三焦"}
    
    # 收集所有匹配
    for formula in FORMULAS:
        if formula in text:
            freq = text.count(formula)
            all_matches.append((formula, "formula", freq, len(formula)))
    
    for syndrome in SYNDROMES:
        if syndrome in text:
            freq = text.count(syndrome)
            all_matches.append((syndrome, "syndrome", freq, len(syndrome)))
    
    for herb in HERBS:
        if herb in text:
            freq = text.count(herb)
            all_matches.append((herb, "herb", freq, len(herb)))
    
    for symptom in SYMPTOMS:
        if symptom in text:
            freq = text.count(symptom)
            all_matches.append((symptom, "symptom", freq, len(symptom)))
    
    # 书本特定主题
    if book == "针灸":
        for topic in ACU_TOPICS:
            if topic in text:
                freq = text.count(topic)
                all_matches.append((topic, "acupuncture", freq, len(topic)))
    elif book == "内经":
        for topic in NEIJING_TOPICS:
            if len(topic) >= 2 and topic in text:
                freq = text.count(topic)
                all_matches.append((topic, "neijing", freq, len(topic)))
    elif book == "本草":
        for topic in BENCAO_TOPICS:
            if topic in text:
                freq = text.count(topic)
                all_matches.append((topic, "bencao", freq, len(topic)))
    elif book == "伤寒":
        for topic in SHANGHAN_TOPICS:
            if topic in text:
                freq = text.count(topic)
                all_matches.append((topic, "shanghan", freq, len(topic)))
    elif book == "金匮":
        for topic in JINKUI_TOPICS:
            if topic in text:
                freq = text.count(topic)
                all_matches.append((topic, "jinkui", freq, len(topic)))
    
    for concept in DIAGNOSIS_CONCEPTS:
        if concept in text:
            freq = text.count(concept)
            all_matches.append((concept, "diagnosis", freq, len(concept)))
    
    for concept in TREATMENT_CONCEPTS:
        if concept in text:
            freq = text.count(concept)
            all_matches.append((concept, "treatment", freq, len(concept)))
    
    # 通用概念（最低优先级）
    general_concepts = [
        "营卫不和", "传经", "六经", "辨证", "经方", "时方",
        "表里", "寒热", "虚实", "脉象",
        "阴阳", "五行", "气血", "津液", "营卫",
    ]
    for concept in general_concepts:
        if concept in text:
            freq = text.count(concept)
            all_matches.append((concept, "concept", freq, len(concept)))
    
    if not all_matches:
        return None
    
    # 分离非通用匹配和通用匹配
    non_common = [m for m in all_matches if m[0] not in VERY_COMMON]
    candidates = non_common if non_common else all_matches
    
    # 排序：长度降序 → 词频降序
    candidates.sort(key=lambda x: (x[3], x[2]), reverse=True)
    
    best = candidates[0]
    return (best[0], best[1])

# 内经脏腑上下文检测
ZANGFU_CONTEXT = {
    "心": {
        "脉": "心主血脉的生理功能和病理表现",
        "神": "心藏神与神志的关系",
        "火": "心在五行属火的理论",
        "夏": "心与夏季养生的关系",
        "舌": "心开窍于舌的理论",
        "汗": "心与汗的关系",
        "肾": "心肾相交的理论",
        "脾": "心脾关系在临床中的意义",
        "肺": "心肺关系在生理病理中的意义",
        "主": "心的主要生理功能",
    },
    "肝": {
        "血": "肝藏血的生理功能",
        "魂": "肝藏魂与神志的关系",
        "木": "肝在五行属木的理论",
        "春": "肝与春季养生的关系",
        "目": "肝开窍于目的理论",
        "肾": "肝肾同源的理论",
        "脾": "肝脾关系在临床中的意义",
        "主": "肝的主要生理功能",
        "疏泄": "肝主疏泄的功能和病理",
    },
    "脾": {
        "血": "脾统血的生理功能",
        "意": "脾藏意与神志的关系",
        "土": "脾在五行属土的理论",
        "长夏": "脾与长夏养生的关系",
        "口": "脾开窍于口的理论",
        "主": "脾的主要生理功能",
        "运化": "脾主运化的功能和病理",
        "肾": "脾肾关系在临床中的意义",
        "心": "心脾关系在临床中的意义",
    },
    "肺": {
        "气": "肺主气的生理功能",
        "魄": "肺藏魄与神志的关系",
        "金": "肺在五行属金的理论",
        "秋": "肺与秋季养生的关系",
        "鼻": "肺开窍于鼻的理论",
        "皮": "肺合皮毛的理论",
        "主": "肺的主要生理功能",
        "肾": "肺肾关系在临床中的意义",
        "通调": "肺通调水道的功能",
    },
    "肾": {
        "精": "肾藏精的生理功能",
        "志": "肾藏志与神志的关系",
        "水": "肾在五行属水的理论",
        "冬": "肾与冬季养生的关系",
        "耳": "肾开窍于耳的理论",
        "骨": "肾主骨生髓的理论",
        "主": "肾的主要生理功能",
        "心": "心肾相交的理论",
        "脾": "脾肾关系在临床中的意义",
        "肺": "肺肾关系（金水相生）的理论",
    },
}

def detect_zangfu_aspect(text: str, zangfu: str) -> str:
    """检测脏腑讨论的具体方面"""
    head = text[:400]
    context_map = ZANGFU_CONTEXT.get(zangfu, {})
    for keyword, description in context_map.items():
        if keyword in head:
            return description
    return f"{zangfu}的主要生理功能和病理表现"


def extract_key_phrase(text: str) -> str | None:
    """
    从文本首句提取关键短语，用于 fallback instruction 生成。
    
    改进策略（v3）：
    1. 多模式提取：主语-谓语、话题-说明、名词短语
    2. 质量验证：通过 validate_key_phrase 检查
    3. 清理标点和停止词
    4. 失败时返回 None（而非低质量短语）
    """
    first_line = text.split('\n')[0].strip()
    # 去掉条文编号前缀
    first_line = re.sub(r'^[一二三四五六七八九十百千]+[：:]\s*', '', first_line)
    # 取前 2 个句子
    sentences = re.split(r'(?<=[。！？])', first_line)
    sentence = sentences[0].strip() if sentences else first_line[:80]
    
    candidates = []
    
    # 模式1: "X是Y" / "X就是Y" -> 提取 X
    m = re.match(r'^(.{2,12})[是就]', sentence)
    if m:
        candidates.append(m.group(1))
    
    # 模式2: "X的Y" -> 提取 "X的Y"
    m = re.match(r'^(.{2,8})的(.{2,6})', sentence)
    if m:
        candidates.append(f"{m.group(1)}的{m.group(2)}")
    
    # 模式3: "讲到X" / "谈到X" / "说X" -> 提取 X
    m = re.search(r'(?:讲到|谈到|说到|提到|讲|谈)(.{2,12})', sentence)
    if m:
        candidates.append(m.group(1))
    
    # 模式4: 包含方剂名/药材名/症状的子串（从已知列表中提取）
    for formula in FORMULAS[:30]:  # 只查前30个最常见方剂
        if formula in sentence:
            candidates.append(formula)
            break
    for herb in HERBS[:30]:
        if herb in sentence:
            candidates.append(herb)
            break
    for symptom in SYMPTOMS[:20]:
        if symptom in sentence:
            candidates.append(symptom)
            break
    
    # 模式5: 提取首个名词性短语（连续中文字符，4-10字）
    # 只在前面模式都没匹配时使用，且只取较短的部分
    m = re.match(r'^([\u4e00-\u9fff]{4,10})', sentence)
    if m:
        candidates.append(m.group(1))
    
    # v3.2: 移除了模式6（原始截取前12字）——质量太低
    
    # 对每个候选进行清理和验证
    for candidate in candidates:
        cleaned = clean_key_phrase(candidate)
        if cleaned and validate_key_phrase(cleaned):
            return cleaned
    
    return None


def generate_refined_instruction(text: str, book: str, source: str = "pdf") -> str:
    """
    精化指令生成。
    
    核心改进：
    1. 主题检测只看前 1-2 句（避免全文关键词干扰）
    2. 内经脏腑用上下文检测具体方面
    3. 方剂/药材/证型用方面检测差异化
    4. Fallback 从首句提取关键短语
    """
    # formulas_db 数据已有精准 instruction，直接保留
    if source == "formulas_db":
        return None  # 调用方保留原 instruction
    
    text = text.strip()
    first_sentences = get_first_sentences(text, n=2)
    head = text[:400]
    
    # 1. 条文编号
    first_line = text.split('\n')[0].strip()
    clause_match = CLAUSE_PATTERN.match(first_line)
    if clause_match:
        clause_num = clause_match.group(1)
        clause_text = clause_match.group(2)[:60]
        book_ref = "伤寒论" if book == "伤寒" else "金匮要略" if book == "金匮" else "中医经典"
        return f"请解释{book_ref}第{clause_num}条：「{clause_text}」"
    
    # 2. 方剂名（在前 2 句中查找——方剂名足够具体，可以在前 2 句匹配）
    formula = find_topic_in_first_sentence(text, FORMULAS)
    if not formula:
        # 如果前 2 句没找到，在前 400 字找（有些段落开头是引文）
        formula = find_topic_in_text(head, FORMULAS, search_range=400)
    if formula:
        aspect = detect_aspect(text, ASPECT_PATTERNS)
        if aspect == "组成":
            return f"请讲解{formula}的药物组成。"
        elif aspect == "主治":
            return f"{formula}主治什么证候？"
        elif aspect == "病机":
            return f"{formula}的组方原理和病机是什么？"
        elif aspect == "加减":
            return f"{formula}有哪些常用的加减变化？"
        elif aspect == "禁忌":
            return f"{formula}的使用禁忌和注意事项是什么？"
        elif aspect == "辨证":
            return f"如何辨证使用{formula}？"
        elif aspect == "鉴别":
            return f"{formula}与相关方剂如何鉴别使用？"
        elif aspect == "剂量":
            return f"{formula}的剂量和煎服法是怎样的？"
        elif aspect == "配伍":
            return f"{formula}中各药物的配伍意义是什么？"
        else:
            return f"请讲解{formula}的组方原理和临床应用。"
    
    # 3. 证型/病名（只在前 2 句中查找——避免宽泛匹配）
    syndrome = find_topic_in_first_sentence(text, SYNDROMES)
    if syndrome:
        aspect = detect_aspect(text, SYNDROME_ASPECTS)
        if aspect == "病机":
            return f"什么是{syndrome}？其病机是什么？"
        elif aspect == "症状":
            return f"{syndrome}的临床表现和辨证要点是什么？"
        elif aspect == "治疗":
            return f"{syndrome}应该用什么方剂治疗？"
        elif aspect == "传变":
            return f"{syndrome}的传变规律和预后如何？"
        elif aspect == "辨证":
            return f"如何辨证{syndrome}？与其他证型如何鉴别？"
        else:
            return f"请讲解{syndrome}的病机和治疗原则。"
    
    # 4. 药材名（只在前 2 句中查找）
    herb = find_topic_in_first_sentence(text, HERBS)
    if herb:
        aspect = detect_aspect(text, ASPECT_PATTERNS)
        if aspect == "配伍":
            return f"{herb}在经方中常与哪些药材配伍？"
        elif aspect == "主治":
            return f"{herb}的主要功效和主治是什么？"
        elif aspect == "病机":
            return f"请讲解{herb}的药性和作用机理。"
        elif aspect == "禁忌":
            return f"{herb}的使用禁忌和注意事项有哪些？"
        elif aspect == "剂量":
            return f"{herb}的常用剂量和炮制方法是什么？"
        else:
            return f"请讲解{herb}在经方中的应用。"
    
    # 5. 按书本选择特定主题
    # 5a. 针灸：经络/穴位（只在前 2 句查找）
    if book == "针灸":
        acu_topic = find_topic_in_first_sentence(text, ACU_TOPICS)
        if acu_topic:
            aspect = detect_aspect(text, ACU_ASPECTS)
            if aspect == "定位":
                return f"请问{acu_topic}的定位和取穴方法是什么？"
            elif aspect == "循行":
                return f"{acu_topic}的循行路线是怎样的？"
            elif aspect == "主治":
                return f"{acu_topic}主治哪些病证？"
            elif aspect == "针刺":
                return f"{acu_topic}的针刺手法和注意事项是什么？"
            elif aspect == "配穴":
                return f"{acu_topic}在临床上常与哪些穴位配伍？"
            else:
                return f"请讲解{acu_topic}的经络理论和临床应用。"
    
    # 5b. 内经：脏腑需要上下文检测
    if book == "内经":
        # 先查具体理论词（非单字脏腑）
        specific_topics = [t for t in NEIJING_TOPICS if len(t) >= 2 and t not in ("心", "肝", "脾", "肺", "肾", "神")]
        neijing_topic = find_topic_in_first_sentence(text, specific_topics)
        if neijing_topic:
            if neijing_topic in ("阴阳", "五行"):
                return f"请讲解《黄帝内经》中{neijing_topic}理论及其临床应用。"
            elif neijing_topic in ("望诊", "闻诊", "问诊", "切诊", "脉诊", "舌诊"):
                return f"请讲解《黄帝内经》中{neijing_topic}的方法和临床意义。"
            elif neijing_topic in ("四气调神", "生气通天", "上古天真"):
                return f"请讲解《黄帝内经·{neijing_topic}篇》的主要内容。"
            else:
                return f"请讲解《黄帝内经》中关于{neijing_topic}的理论。"
        
        # 再查单字脏腑（需要上下文检测）
        zangfu = find_topic_in_first_sentence(text, ["心", "肝", "脾", "肺", "肾"])
        if zangfu:
            aspect_desc = detect_zangfu_aspect(text, zangfu)
            return f"请讲解《黄帝内经》中关于{aspect_desc}。"
    
    # 5c. 本草
    if book == "本草":
        # 先查药材名
        herb_topic = find_topic_in_first_sentence(text, HERBS)
        if herb_topic:
            aspect = detect_aspect(text, ASPECT_PATTERNS)
            if aspect == "病机":
                return f"请讲解{herb_topic}的药性理论和作用机理。"
            elif aspect == "配伍":
                return f"请讲解{herb_topic}在方剂配伍中的应用。"
            else:
                return f"请讲解{herb_topic}的功效和临床应用。"
        # 再查本草理论
        bencao_topic = find_topic_in_first_sentence(text, BENCAO_TOPICS)
        if bencao_topic:
            return f"请讲解中药学中{bencao_topic}的理论和临床应用。"
    
    # 5d. 伤寒
    if book == "伤寒":
        shanghan_topic = find_topic_in_first_sentence(text, SHANGHAN_TOPICS)
        if shanghan_topic:
            return f"请讲解伤寒论中{shanghan_topic}的理论和临床意义。"
    
    # 5e. 金匮
    if book == "金匮":
        jinkui_topic = find_topic_in_first_sentence(text, JINKUI_TOPICS)
        if jinkui_topic:
            return f"请讲解金匮要略中{jinkui_topic}的辨治要点。"
    
    # 6. 通用概念检测（在前 2 句中查找）
    concept = find_topic_in_first_sentence(text, [
        "营卫不和", "传经", "六经", "辨证", "经方", "时方",
        "表里", "寒热", "虚实", "脉象",
        "恶寒", "恶风", "发热", "汗出", "无汗", "头痛",
        "体痛", "呕逆", "下利", "烦躁", "口渴", "心下痞",
        "但热不寒", "往来寒热", "四肢厥逆",
    ])
    if concept:
        return f"请解释{concept}的含义和临床意义。"
    
    # 6.5 全文医学术语扫描（首句未匹配时，扫描全文找医学词）
    # 这是 v3 的核心改进：大部分 fallback 数据实际上包含医学词，
    # 只是没出现在前 1-2 句中（倪师讲课风格：先讲案例再引入主题）
    scan_result = scan_text_for_medical_term(text, book)
    if scan_result:
        term, category = scan_result
        aspect = detect_aspect(text, ASPECT_PATTERNS)
        if category == "formula":
            if aspect == "组成":
                return f"请讲解{term}的药物组成。"
            elif aspect == "主治":
                return f"{term}主治什么证候？"
            elif aspect == "病机":
                return f"{term}的组方原理和病机是什么？"
            elif aspect == "加减":
                return f"{term}有哪些常用的加减变化？"
            elif aspect == "禁忌":
                return f"{term}的使用禁忌和注意事项是什么？"
            elif aspect == "配伍":
                return f"{term}中各药物的配伍意义是什么？"
            else:
                return f"请讲解{term}的组方原理和临床应用。"
        elif category == "syndrome":
            syn_aspect = detect_aspect(text, SYNDROME_ASPECTS)
            if syn_aspect == "病机":
                return f"什么是{term}？其病机是什么？"
            elif syn_aspect == "症状":
                return f"{term}的临床表现和辨证要点是什么？"
            elif syn_aspect == "治疗":
                return f"{term}应该用什么方剂治疗？"
            else:
                return f"请讲解{term}的病机和治疗原则。"
        elif category == "herb":
            if aspect == "配伍":
                return f"{term}在经方中常与哪些药材配伍？"
            elif aspect == "主治":
                return f"{term}的主要功效和主治是什么？"
            elif aspect == "禁忌":
                return f"{term}的使用禁忌和注意事项有哪些？"
            else:
                return f"请讲解{term}在经方中的应用。"
        elif category == "symptom":
            return f"请解释{term}的临床意义和辨证价值。"
        elif category == "acupuncture":
            acu_aspect = detect_aspect(text, ACU_ASPECTS)
            if acu_aspect == "定位":
                return f"请问{term}的定位和取穴方法是什么？"
            elif acu_aspect == "循行":
                return f"{term}的循行路线是怎样的？"
            elif acu_aspect == "主治":
                return f"{term}主治哪些病证？"
            else:
                return f"请讲解{term}的经络理论和临床应用。"
        elif category == "neijing":
            if term in ("阴阳", "五行"):
                return f"请讲解《黄帝内经》中{term}理论及其临床应用。"
            elif term in ("心", "肝", "脾", "肺", "肾"):
                aspect_desc = detect_zangfu_aspect(text, term)
                return f"请讲解《黄帝内经》中关于{aspect_desc}。"
            else:
                return f"请讲解《黄帝内经》中关于{term}的理论。"
        elif category == "bencao":
            return f"请讲解中药学中{term}的理论和临床应用。"
        elif category == "shanghan":
            return f"请讲解伤寒论中{term}的理论和临床意义。"
        elif category == "jinkui":
            return f"请讲解金匮要略中{term}的辨治要点。"
        elif category == "diagnosis":
            return f"请讲解{term}的方法和临床意义。"
        elif category == "treatment":
            return f"请讲解{term}的治则和临床应用。"
        else:
            return f"请解释{term}的含义和临床意义。"
    
    # 7. 智能关键短语提取（带质量验证，v3 大幅改进）
    key_phrase = extract_key_phrase(text)
    if key_phrase:
        return f"请讲解关于「{key_phrase}」的中医知识。"
    
    # 8. 终极 Fallback（应极少触发，v3 改进：尝试从文本中提取任何有意义短语）
    book_desc = {
        "针灸": "针灸经络穴位的临床应用",
        "内经": "黄帝内经的中医基础理论",
        "本草": "中药药性理论和临床应用",
        "伤寒": "伤寒论六经辨证的理论",
        "金匮": "金匮要略杂病辨治的方法",
    }
    desc = book_desc.get(book, "中医基础理论")
    # 最后一次尝试：从文本前 200 字中提取有意义短语
    for pattern in [r'讲到(.{2,8})', r'关于(.{2,8})', r'(.{2,6})的(.{2,6})']:
        m = re.search(pattern, text[:200])
        if m:
            if m.lastindex == 1:
                phrase = clean_key_phrase(m.group(1))
            else:
                phrase = clean_key_phrase(f"{m.group(1)}的{m.group(2)}")
            if phrase and validate_key_phrase(phrase):
                return f"请讲解关于「{phrase}」的中医知识。"
    return f"请讲解{desc}。"


# ============================================================
# 5. 去重
# ============================================================

def deduplicate(data: list[dict]) -> list[dict]:
    """基于 output 内容去重"""
    seen = set()
    unique = []
    for d in data:
        # 用 output 前 150 字做指纹
        key = d["output"][:150]
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


# ============================================================
# 6. 主流程
# ============================================================

def process_pdf_data(input_path: str) -> list[dict]:
    """处理 PDF 提取的原始数据"""
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(l) for l in f]
    
    print(f"输入: {len(raw_data)} 条原始 PDF 数据")
    
    processed = []
    stats = Counter()
    
    for d in raw_data:
        text = d["output"]
        book = d.get("book", "")
        
        # Step 1: 噪音清洗（英文+集数标记）
        text = clean_noise(text)
        if not text or len(text) < 50:
            stats["noise_cleaned_empty"] += 1
            continue
        
        # Step 2: 非教学内容过滤（生平/目录/易筋经等）
        if is_non_teaching(text):
            stats["non_teaching_filtered"] += 1
            continue
        
        # Step 3: 上下文断裂修复（代词开头）
        text = fix_context_break(text)
        if text is None:
            stats["context_break_filtered"] += 1
            continue
        
        # Step 4: 长度控制（<200 过滤，>800 切分）
        chunks = control_length(text, min_len=200, max_len=800)
        if not chunks:
            stats["too_short_filtered"] += 1
            continue
        
        for chunk in chunks:
            # Step 5: 再次检查非教学（切分后可能暴露噪音）
            if is_non_teaching(chunk):
                stats["post_split_non_teaching"] += 1
                continue
            
            # Step 6: 指令精化
            instruction = generate_refined_instruction(chunk, book, source="pdf")
            
            processed.append({
                "instruction": instruction,
                "input": "",
                "output": chunk,
                "book": book,
                "source": "pdf",
            })
            stats["processed"] += 1
    
    print(f"处理后: {len(processed)} 条")
    print(f"统计: {dict(stats)}")
    return processed


def load_db_data(input_path: str) -> list[dict]:
    """
    加载 formulas_db 知识数据（保留原 instruction，过滤过短条目）。
    对短输出条目添加 quality 标记。
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(l) for l in f]
    
    processed = []
    filtered_short = 0
    quality_stats = Counter()
    for d in raw_data:
        output = d.get("output", "")
        # 过滤过短条目（< 30 字信息量不足）
        if len(output) < 30:
            filtered_short += 1
            continue
        
        entry = {
            "instruction": d["instruction"],
            "input": d.get("input", ""),
            "output": output,
            "book": "伤寒",
            "source": "formulas_db",
            "category": d.get("category", ""),
        }
        
        # 质量标记
        if len(output) < 50:
            entry["quality"] = "short"
            quality_stats["short"] += 1
        elif len(output) < 100:
            entry["quality"] = "medium"
            quality_stats["medium"] += 1
        else:
            entry["quality"] = "good"
            quality_stats["good"] += 1
        
        processed.append(entry)
    
    print(f"formulas_db 数据: {len(processed)} 条 (过滤 {filtered_short} 条过短)")
    print(f"  质量分布: {dict(quality_stats)}")
    return processed


def mix_data(pdf_data: list[dict], db_data: list[dict], pdf_ratio: float = 0.7) -> list[dict]:
    """
    混合 PDF 风格数据和 formulas_db 知识数据。
    pdf_ratio: PDF 数据占比（默认 70%）
    """
    # 计算需要的 formulas_db 条数
    target_db = int(len(pdf_data) * (1 - pdf_ratio) / pdf_ratio)
    
    # 如果 formulas_db 数据不够，全部用上
    if len(db_data) <= target_db:
        selected_db = db_data
    else:
        # 随机采样
        random.seed(42)
        selected_db = random.sample(db_data, target_db)
    
    mixed = pdf_data + selected_db
    random.seed(42)
    random.shuffle(mixed)
    
    print(f"混合: PDF {len(pdf_data)} 条 + formulas_db {len(selected_db)} 条 = {len(mixed)} 条")
    return mixed


def split_dataset(data: list[dict], val_ratio: float = 0.1) -> tuple[list[dict], list[dict]]:
    """划分 train/val"""
    random.seed(42)
    random.shuffle(data)
    val_size = int(len(data) * val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]
    return train_data, val_data


def generate_report(train_data: list[dict], val_data: list[dict], stats: Counter) -> dict:
    """生成质量报告"""
    all_data = train_data + val_data
    
    # Instruction 唯一性
    instructions = [d["instruction"] for d in all_data]
    unique_instructions = len(set(instructions))
    
    # 口语标记
    oral_markers = ["什么意思呢", "你想想看", "诸位", "就是说", "所以说", "好不好",
                    "简单讲", "实际上", "就这样子", "你错了", "我告诉", "临床",
                    "我们在", "为什么", "意思就是", "对不对", "了解", "知道"]
    oral_count = sum(1 for d in all_data if any(m in d["output"] for m in oral_markers))
    
    # 长度分布
    lengths = [len(d["output"]) for d in all_data]
    
    # 按来源
    source_counts = Counter(d.get("source", "unknown") for d in all_data)
    
    # 按书
    book_counts = Counter(d.get("book", "?") for d in all_data)
    
    # Instruction 重复分析
    inst_counter = Counter(instructions)
    top_dups = inst_counter.most_common(10)
    
    # 残留噪音检查
    english_remaining = sum(1 for d in all_data if re.findall(r'[a-zA-Z]{3,}', d["output"]))
    episode_remaining = sum(1 for d in all_data if re.search(r'\[第\d+\s*集\]', d["output"]))
    
    return {
        "total": len(all_data),
        "train": len(train_data),
        "val": len(val_data),
        "unique_instructions": unique_instructions,
        "unique_instruction_ratio": f"{unique_instructions}/{len(all_data)} ({unique_instructions/len(all_data)*100:.1f}%)",
        "oral_coverage": f"{oral_count}/{len(all_data)} ({oral_count/len(all_data)*100:.1f}%)",
        "length": {
            "min": min(lengths), "max": max(lengths), "avg": sum(lengths)//len(lengths),
            "200_500": sum(1 for l in lengths if 200 <= l < 500),
            "500_800": sum(1 for l in lengths if 500 <= l < 800),
            "800_plus": sum(1 for l in lengths if l >= 800),
        },
        "by_source": dict(source_counts),
        "by_book": dict(book_counts),
        "top_instruction_dups": [(inst, count) for inst, count in top_dups],
        "residual_english": english_remaining,
        "residual_episode_markers": episode_remaining,
        "processing_stats": dict(stats),
    }


def main():
    print("=" * 60)
    print("SFT 数据精炼管道")
    print("=" * 60)
    
    # 处理 PDF 数据
    pdf_data = process_pdf_data(str(INPUT_PDF))
    
    # 去重
    pdf_data = deduplicate(pdf_data)
    print(f"PDF 去重后: {len(pdf_data)} 条")
    
    # 不混入 formulas_db 数据（正式结构化文本会稀释口语化信号）
    # 知识查询能力由 GraphRAG 提供，SFT 只负责口语化风格迁移
    mixed = pdf_data
    
    # 划分
    train_data, val_data = split_dataset(mixed, val_ratio=0.1)
    
    # 输出
    train_path = OUTPUT_DIR / "sft_train_final.jsonl"
    val_path = OUTPUT_DIR / "sft_val_final.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for d in train_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for d in val_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    print(f"\n训练集: {train_path} ({len(train_data)} 条)")
    print(f"验证集: {val_path} ({len(val_data)} 条)")
    
    # 质量报告
    report = generate_report(train_data, val_data, Counter())
    report_path = OUTPUT_DIR / "sft_refine_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("质量报告:")
    print(f"  总条数: {report['total']} (train {report['train']} + val {report['val']})")
    print(f"  唯一 instruction: {report['unique_instruction_ratio']}")
    print(f"  口语覆盖率: {report['oral_coverage']}")
    print(f"  长度: avg {report['length']['avg']}, 范围 {report['length']['min']}-{report['length']['max']}")
    print(f"  按来源: {report['by_source']}")
    print(f"  按书: {report['by_book']}")
    print(f"  残留英文: {report['residual_english']}")
    print(f"  残留集数标记: {report['residual_episode_markers']}")
    print(f"\n  Instruction 重复 TOP 5:")
    for inst, count in report['top_instruction_dups'][:5]:
        print(f"    [{count}次] {inst[:70]}")
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
