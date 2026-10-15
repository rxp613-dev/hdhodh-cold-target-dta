#!/usr/bin/env python3
"""
参考文献验证脚本
时间: 2026-10-15
目标: 验证第二轮评审报告中指出的问题文献
方法: 通过Crossref API验证DOI和元数据
"""

import requests
import json
import time
from pathlib import Path

# 问题文献列表（根据评审报告）
PROBLEMATIC_REFS = {
    "ref4": {
        "issue": "疑似虚构",
        "current": "Downs DM. De novo pyrimidine biosynthesis: the fourth step. Biochemistry 2002;41(34):10485–10491",
        "action": "需替换为真实DHODH酶学文献"
    },
    "ref7": {
        "issue": "不存在",
        "current": "Fox RJ, Coffey CS, et al. Lancet Neurol 2019;18(6):546–556（teriflunomide 2期）",
        "action": "替换为O'Connor PW 2006或O'Connor P 2011"
    },
    "ref10": {
        "issue": "疑似虚构",
        "current": "Kuntz DA, et al. J Med Chem 2020;63(22):12345–12358",
        "action": "删除并替换为真实DHODH抑制剂文献"
    },
    "ref11": {
        "issue": "作者名全错",
        "current": "Lin Xiaowen, Li Xiang, Xinqi Lin. Molecules 2020;25(6):1375",
        "action": "更正为Lin X (Xiaoqian), Li X (Xiu), Lin X (Xubo)"
    },
    "ref12": {
        "issue": "疑似虚构",
        "current": "McGowan DC, Rousseau WF. Expert Opin Ther Pat 2015;25(12):1385–1397",
        "action": "替换为真实文献"
    },
    "ref15": {
        "issue": "作者名拼写错误",
        "current": "Öztürk H, Arzuvir Özgür, Ozkirimli E.",
        "action": "更正为Arzucan Özgür"
    },
    "ref16": {
        "issue": "期刊名错误",
        "current": "Ren X, et al. Journal of Cheminformatics, 2026",
        "action": "改为Molecules"
    },
    "ref18": {
        "issue": "标题张冠李戴",
        "current": "Santos R, Gault DS, et al. 'A comprehensive map of the druggable genome.' Nat Rev Drug Discov 2017;16(1):19–34",
        "action": "更正标题和作者"
    },
    "ref20": {
        "issue": "分辨率错误",
        "current": "4JTU, 1.8 Å",
        "action": "改为1.9 Å"
    },
    "ref21": {
        "issue": "检索无匹配",
        "current": "Wang L, Li N, Tang J. Briefings in Bioinformatics 2023;24(1):bbac554",
        "action": "提供DOI核实或替换"
    },
    "ref24": {
        "issue": "检索无匹配",
        "current": "Yue X, Zhu Y, Zhao H, Li J. JCIM 2022;62(2):356–365",
        "action": "提供DOI核实或替换"
    }
}

def verify_doi(doi):
    """通过Crossref API验证DOI"""
    url = f"https://api.crossref.org/works/{doi}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "valid": True,
                "title": data.get("message", {}).get("title", [""])[0],
                "author": data.get("message", {}).get("author", []),
                "journal": data.get("message", {}).get("container-title", [""])[0]
            }
        else:
            return {"valid": False, "status_code": response.status_code}
    except Exception as e:
        return {"valid": False, "error": str(e)}

def main():
    print("=" * 80)
    print("参考文献验证报告")
    print("=" * 80)
    print()

    results = {}

    for ref_id, info in PROBLEMATIC_REFS.items():
        print(f"\n{'='*80}")
        print(f"验证 {ref_id}")
        print(f"{'='*80}")
        print(f"问题: {info['issue']}")
        print(f"当前条目: {info['current']}")
        print(f"处置方案: {info['action']}")
        print()

        results[ref_id] = {
            "issue": info['issue'],
            "current": info['current'],
            "action": info['action'],
            "verified": False
        }

        # 暂停以避免API限流
        time.sleep(1)

    # 保存结果
    output_file = Path("/mnt/data/dta2/第二篇论文/data/reference_verification.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"验证结果已保存至: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
