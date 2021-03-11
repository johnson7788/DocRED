# DocRED
论文的代码 [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127v3)


文档中的多个实体通常表现出复杂的句间关系，现有的关系抽取（RE）方法无法很好地处理，因为现有的关系提取（RE）方法通常侧重于提取单个实体对的句内关系。
为了加快文档级RE的研究，我们引入了DocRED，这是一个由Wikipedia和Wikidata构建的具有三个特征的新数据集。

+ DocRED对命名的实体和关系进行标注，是最大的从纯文本中提取文档级RE的人工标注数据集。
+ DocRED需要阅读文档中的多个句子来抽取实体，并通过综合文档的所有信息来推理它们的关系。
+ 除了人工标注的数据，我们还提供了大规模的远端有监督数据，这使得DocRED可以同时适用于监督和弱监督的场景。


## Codalab
If you are interested in our dataset, you are welcome to join in the Codalab competition at [DocRED](https://competitions.codalab.org/competitions/20717)


## Cite
If you use the dataset or the code, please cite this paper:
```
@inproceedings{yao2019DocRED,
  title={{DocRED}: A Large-Scale Document-Level Relation Extraction Dataset},
  author={Yao, Yuan and Ye, Deming and Li, Peng and Han, Xu and Lin, Yankai and Liu, Zhenghao and Liu, Zhiyuan and Huang, Lixin and Zhou, Jie and Sun, Maosong},
  booktitle={Proceedings of ACL 2019},
  year={2019}
}
```
