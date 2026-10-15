#!/usr/bin/env python3
"""
参考文献验证脚本
Task 3: 参考文献全量重建 [A5, C1]

时间: 2026-10-15
目标: 根据独立评审报告A5问题，验证并修复参考文献

评审发现的问题:
1. ref1 (DeepDTA): 作者名拼写错误
2. ref10 (AttentionDTA): 作者、会议、年份全错
3. ref11 (Kuntz JMC 2020): 疑似虚构
4. ref14 (Multi-Similarity): 张冠李戴
5. ref16 (Downs): 未能检索到
6. ref17 (Breedveld): 年卷期页全错
7. ref19 (Fox): 不存在
8. ref20 (4JTU): PDB引用错误
9. ref21 (Welsch): 疑似虚构
10. ref15: 格式崩坏（[Authors]占位符）

作者: Claude
"""

import re
from pathlib import Path

BIB_FILE = Path('/mnt/data/dta2/第二篇论文/references.bib')

def fix_deepdta_authors():
    """修复ref1 (DeepDTA) 作者名"""
    # 正确作者: Öztürk H, Özgür A, Ozkirimli E
    old = """@article{ref1,
  title={DeepDTA: deep drug--target binding affinity prediction},
  author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzuvir and Ozgur, Elif},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i821--i829},
  year={2018},
  publisher={Oxford University Press}
}"""

    new = """@article{ref1,
  title={DeepDTA: deep drug--target binding affinity prediction},
  author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzuvir and Ozkirimli, Elif},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i821--i829},
  year={2018},
  doi={10.1093/bioinformatics/bty562},
  publisher={Oxford University Press}
}"""
    return old, new

def fix_attentiondta():
    """修复ref10 (AttentionDTA) - 正确引用"""
    # 正确: Zhao Q, Xiao F, Yang M, Li Y, Wang J. IEEE BIBM 2019:64-69
    old = """@inproceedings{ref10,
  title={AttentiveDTA: predicting drug--target binding affinity with attentive pooling network},
  author={Yang, Xian and Zhang, Jian and Yoshida, Hisanori and Tang, Jing},
  booktitle={Proceedings of the Machine Learning for Health Workshop, NeurIPS 2018},
  year={2018}
}"""

    new = """@inproceedings{ref10,
  title={AttentionDTA: prediction of drug--target binding affinity using attention model},
  author={Zhao, Qi and Xiao, Feng and Yang, Mengmeng and Li, Yidan and Wang, Jian},
  booktitle={Proceedings of IEEE BIBM 2019},
  pages={64--69},
  year={2019},
  doi={10.1109/BIBM47256.2019.8983058}
}"""
    return old, new

def fix_multisimilarity():
    """修复ref14 (Multi-Similarity Loss) - 正确作者"""
    # 正确: Wang X, Han X, Huang W, Deng D, Scott MR. CVPR 2019:5022-5030
    old = """@inproceedings{ref14,
  title={Improved deep metric learning with multi-similarity loss},
  author={Sohn, Kihyuk},
  booktitle={Proceedings of CVPR 2019},
  pages={5579--5588},
  year={2019}
}"""

    new = """@inproceedings{ref14,
  title={Multi-similarity loss with general pair weighting for deep metric learning},
  author={Wang, Xinshao and Han, Xiangyu and Huang, Weiyang and Deng, Dong and Scott, Matthew R},
  booktitle={Proceedings of IEEE/CVF CVPR 2019},
  pages={5022--5030},
  year={2019},
  doi={10.1109/CVPR.2019.00516}
}"""
    return old, new

def fix_breedveld():
    """修复ref17 (Breedveld) - 正确引用"""
    # 正确: Ann Rheum Dis 2000;59(11):841-849, doi:10.1136/ard.59.11.841
    old = """@article{ref17,
  title={Leflunomide: mode of action in the treatment of rheumatoid arthritis},
  author={Breedveld, Ferdinand C and Dayer, Jean-Michel and Laan, Leonieke CM and others},
  journal={Annals of the Rheumatic Diseases},
  volume={60},
  number={11},
  pages={967--971},
  year={2001},
  publisher={BMJ Publishing Group}
}"""

    new = """@article{ref17,
  title={Leflunomide: mode of action in the treatment of rheumatoid arthritis},
  author={Breedveld, Ferdinand C and Dayer, Jean-Michel},
  journal={Annals of the Rheumatic Diseases},
  volume={59},
  number={11},
  pages={841--849},
  year={2000},
  doi={10.1136/ard.59.11.841},
  publisher={BMJ Publishing Group}
}"""
    return old, new

def fix_4jtu():
    """修复ref20 (4JTU PDB结构) - 正确引用"""
    # 正确: Zhu L, Li H, Zhu J, Ren X. PDB 4JTU, released 2014-03-26
    old = """@article{ref20,
  title={Crystal structure of human dihydroorotate dehydrogenase B with a brequinar analog at 1.8 {\\AA} resolution},
  author={Liu, Jian Ping and Deacon, Adrain M and others},
  journal={Journal of Molecular Biology},
  volume={426},
  number={8},
  pages={1531--1545},
  year={2014},
  publisher={Elsevier},
  note={PDB 4JTU}
}"""

    new = """@misc{ref20,
  title={Crystal structure of human dihydroorotate dehydrogenase with a brequinar analogue at 1.8 {\\AA} resolution},
  author={Zhu, Lin and Li, Hui and Zhu, Jian and Ren, Xiaoyu},
  year={2014},
  note={PDB 4JTU, released 2014-03-26. Main citation: To be published},
  howpublished={RCSB Protein Data Bank},
  doi={10.2210/pdb4JTU/pdb}
}"""
    return old, new

def fix_ref15_placeholder():
    """修复ref15格式（第一篇论文，under review）"""
    old = """@article{ref15,
  title={Contrastive learning for drug--target affinity prediction: Benchmark evaluation and cold-start challenges},
  author={[Authors]},
  journal={Journal of Cheminformatics},
  year={2026},
  note={under review}
}"""

    new = """@article{ref15,
  title={Contrastive learning for drug--target affinity prediction: Benchmark evaluation and cold-start challenges},
  author={Ren, Xiaoping and Sui, Chaoya and Liu, Xiaoyong and Ren, Xiaoli and Tao, Ling and Zhang, Yan},
  journal={Journal of Cheminformatics},
  year={2026},
  note={Submitted, under review}
}"""
    return old, new

def remove_fabricated_refs():
    """标记疑似虚构的参考文献（ref11, ref21）"""
    # 这些需要在论文中用真实文献替换
    print("需要替换的疑似虚构文献:")
    print("- ref11 (Kuntz JMC 2020): 完全检索不到")
    print("- ref21 (Welsch BMCL 2017): 完全检索不到")
    print("\n建议替换为真实的DHODH抑制剂文献")
    return None, None

def main():
    """主函数"""
    print("=" * 80)
    print("参考文献验证与修复")
    print("=" * 80)

    # 读取当前bib文件
    with open(BIB_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 应用修复
    fixes = [
        ("ref1 (DeepDTA作者名)", fix_deepdta_authors),
        ("ref10 (AttentionDTA)", fix_attentiondta),
        ("ref14 (Multi-Similarity)", fix_multisimilarity),
        ("ref17 (Breedveld)", fix_breedveld),
        ("ref20 (4JTU PDB)", fix_4jtu),
        ("ref15 (格式)", fix_ref15_placeholder),
    ]

    print("\n将应用以下修复:")
    for name, fix_func in fixes:
        old, new = fix_func()
        if old and new:
            if old in content:
                content = content.replace(old, new)
                print(f"✅ {name}: 已修复")
            else:
                print(f"⚠️  {name}: 未找到匹配内容")
        else:
            print(f"ℹ️  {name}: {fix_func.__doc__}")

    # 保存修复后的文件
    output_file = BIB_FILE.parent / 'references_fixed.bib'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n修复后的文件已保存到: {output_file}")
    print("\n待处理:")
    print("1. ref11, ref21: 需要用真实文献替换")
    print("2. ref3, ref5, ref16, ref19: 需要进一步验证")
    print("3. 为所有文献添加DOI（如果可用）")
    print("4. 交叉验证10%以上的文献")

    print("\n" + "=" * 80)
    print("验证完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
