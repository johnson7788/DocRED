# Data

Data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw?usp=sharing).

Relation information file has been uploaded.

```buildoutcfg
    ├── dev.json    #开发集
    ├── rel_info.json #label中的关系对应的真正的名字， 例如"P6": "head of government"， 
    ├── test.json   #测试集
    ├── train_annotated.json   #人工标注的数据集，很准确，数据量小
    └── train_distant.json     #远端监督数据集，有噪音，不太准，数据量大
```

```
Data Format:
{
  'title',  文章的标题
  'sents':     [
                  [句子0中的所有单词],
                  [word in sent 1]
               ]
  'vertexSet': [
                  [
                    { 'name': 实体提及的名字，即一个实体, 
                      'sent_id': 实体在某个句子中出现，这个句子的id, 
                      'pos': 实体在这个句子中的起始和结束位置, 
                      'type': 实体的NER类型NER_type
                    }
                    {其它提及}
                  ], 
                  [其它实体]
                
  'labels':   [
                {
                  'h': 第一个实体在vertexSet中的的索引位置
                  't': 第二个实体在vertexSet中的的索引位置,
                  'r': relation, 训练集中使用使用的是P6，代表的是rel_info.json中的"head of government"， 
                  'evidence': 支持证据的句子，支持这个关系的句子的id，支持2个实体的关系的佐证的句子
                }
              ]
}
```


```buildoutcfg
一条train_annotated.json的内容
{
  "vertexSet": [
    [
      {
        "pos": [
          0,
          4
        ],
        "type": "ORG",
        "sent_id": 0,
        "name": "Zest Airways, Inc."
      },
      {
        "sent_id": 0,
        "type": "ORG",
        "pos": [
          10,
          15
        ],
        "name": "Asian Spirit and Zest Air"
      },
      {
        "name": "AirAsia Zest",
        "pos": [
          6,
          8
        ],
        "sent_id": 0,
        "type": "ORG"
      },
      {
        "name": "AirAsia Zest",
        "pos": [
          19,
          21
        ],
        "sent_id": 6,
        "type": "ORG"
      }
    ],
    [
      {
        "name": "Ninoy Aquino International Airport",
        "pos": [
          4,
          8
        ],
        "sent_id": 3,
        "type": "LOC"
      },
      {
        "name": "Ninoy Aquino International Airport",
        "pos": [
          26,
          30
        ],
        "sent_id": 0,
        "type": "LOC"
      }
    ],
    [
      {
        "name": "Pasay City",
        "pos": [
          31,
          33
        ],
        "sent_id": 0,
        "type": "LOC"
      }
    ],
    [
      {
        "name": "Metro Manila",
        "pos": [
          34,
          36
        ],
        "sent_id": 0,
        "type": "LOC"
      }
    ],
    [
      {
        "name": "Philippines",
        "pos": [
          38,
          39
        ],
        "sent_id": 0,
        "type": "LOC"
      },
      {
        "name": "Philippines",
        "pos": [
          13,
          14
        ],
        "sent_id": 4,
        "type": "LOC"
      },
      {
        "sent_id": 5,
        "type": "LOC",
        "pos": [
          25,
          29
        ],
        "name": "Republic of the Philippines"
      }
    ],
    [
      {
        "name": "Manila",
        "pos": [
          13,
          14
        ],
        "sent_id": 1,
        "type": "LOC"
      },
      {
        "name": "Manila",
        "pos": [
          9,
          10
        ],
        "sent_id": 3,
        "type": "LOC"
      }
    ],
    [
      {
        "name": "Cebu",
        "pos": [
          15,
          16
        ],
        "sent_id": 1,
        "type": "LOC"
      }
    ],
    [
      {
        "pos": [
          17,
          18
        ],
        "type": "NUM",
        "sent_id": 1,
        "name": "24"
      }
    ],
    [
      {
        "pos": [
          1,
          2
        ],
        "type": "TIME",
        "sent_id": 2,
        "name": "2013"
      },
      {
        "pos": [
          1,
          5
        ],
        "type": "TIME",
        "sent_id": 5,
        "name": "August 16, 2013"
      }
    ],
    [
      {
        "pos": [
          9,
          11
        ],
        "type": "ORG",
        "name": "Philippines AirAsia",
        "sent_id": 2
      }
    ],
    [
      {
        "pos": [
          5,
          7
        ],
        "type": "ORG",
        "sent_id": 4,
        "name": "Asian Spirit"
      }
    ],
    [
      {
        "pos": [
          7,
          13
        ],
        "type": "ORG",
        "sent_id": 5,
        "name": "Civil Aviation Authority of the Philippines"
      },
      {
        "name": "CAAP",
        "pos": [
          14,
          15
        ],
        "sent_id": 5,
        "type": "ORG"
      }
    ],
    [
      {
        "name": "Zest Air",
        "pos": [
          34,
          36
        ],
        "sent_id": 5,
        "type": "ORG"
      },
      {
        "pos": [
          7,
          9
        ],
        "type": "ORG",
        "sent_id": 6,
        "name": "Zest Air"
      }
    ],
    [
      {
        "sent_id": 6,
        "type": "NUM",
        "pos": [
          2,
          4
        ],
        "name": "a year"
      }
    ],
    [
      {
        "name": "AirAsia",
        "pos": [
          5,
          6
        ],
        "sent_id": 6,
        "type": "ORG"
      }
    ],
    [
      {
        "pos": [
          5,
          7
        ],
        "type": "ORG",
        "name": "AirAsia Philippines",
        "sent_id": 7
      }
    ],
    [
      {
        "pos": [
          8,
          10
        ],
        "type": "TIME",
        "sent_id": 7,
        "name": "January 2016"
      }
    ]
  ],
  "labels": [
    {
      "r": "P159",
      "h": 0,
      "t": 2,
      "evidence": [
        0
      ]
    },
    {
      "r": "P17",
      "h": 0,
      "t": 4,
      "evidence": [
        2,
        4,
        7
      ]
    },
    {
      "r": "P17",
      "h": 12,
      "t": 4,
      "evidence": [
        6,
        7
      ]
    },
    {
      "r": "P17",
      "h": 2,
      "t": 4,
      "evidence": [
        0
      ]
    },
    {
      "r": "P131",
      "h": 2,
      "t": 3,
      "evidence": [
        0
      ]
    },
    {
      "r": "P150",
      "h": 4,
      "t": 3,
      "evidence": [
        0
      ]
    },
    {
      "r": "P17",
      "h": 5,
      "t": 4,
      "evidence": [
        0,
        3
      ]
    },
    {
      "r": "P150",
      "h": 3,
      "t": 2,
      "evidence": [
        0
      ]
    },
    {
      "r": "P131",
      "h": 3,
      "t": 4,
      "evidence": [
        0,
        3
      ]
    },
    {
      "r": "P17",
      "h": 3,
      "t": 4,
      "evidence": [
        0,
        3
      ]
    },
    {
      "r": "P131",
      "h": 1,
      "t": 2,
      "evidence": [
        0,
        3
      ]
    },
    {
      "r": "P17",
      "h": 1,
      "t": 4,
      "evidence": [
        0,
        3
      ]
    },
    {
      "r": "P17",
      "h": 10,
      "t": 4,
      "evidence": [
        4
      ]
    }
  ],
  "title": "AirAsia Zest",
  "sents": [
    [
      "Zest",
      "Airways",
      ",",
      "Inc.",
      "operated",
      "as",
      "AirAsia",
      "Zest",
      "(",
      "formerly",
      "Asian",
      "Spirit",
      "and",
      "Zest",
      "Air",
      ")",
      ",",
      "was",
      "a",
      "low",
      "-",
      "cost",
      "airline",
      "based",
      "at",
      "the",
      "Ninoy",
      "Aquino",
      "International",
      "Airport",
      "in",
      "Pasay",
      "City",
      ",",
      "Metro",
      "Manila",
      "in",
      "the",
      "Philippines",
      "."
    ],
    [
      "It",
      "operated",
      "scheduled",
      "domestic",
      "and",
      "international",
      "tourist",
      "services",
      ",",
      "mainly",
      "feeder",
      "services",
      "linking",
      "Manila",
      "and",
      "Cebu",
      "with",
      "24",
      "domestic",
      "destinations",
      "in",
      "support",
      "of",
      "the",
      "trunk",
      "route",
      "operations",
      "of",
      "other",
      "airlines",
      "."
    ],
    [
      "In",
      "2013",
      ",",
      "the",
      "airline",
      "became",
      "an",
      "affiliate",
      "of",
      "Philippines",
      "AirAsia",
      "operating",
      "their",
      "brand",
      "separately",
      "."
    ],
    [
      "Its",
      "main",
      "base",
      "was",
      "Ninoy",
      "Aquino",
      "International",
      "Airport",
      ",",
      "Manila",
      "."
    ],
    [
      "The",
      "airline",
      "was",
      "founded",
      "as",
      "Asian",
      "Spirit",
      ",",
      "the",
      "first",
      "airline",
      "in",
      "the",
      "Philippines",
      "to",
      "be",
      "run",
      "as",
      "a",
      "cooperative",
      "."
    ],
    [
      "On",
      "August",
      "16",
      ",",
      "2013",
      ",",
      "the",
      "Civil",
      "Aviation",
      "Authority",
      "of",
      "the",
      "Philippines",
      "(",
      "CAAP",
      ")",
      ",",
      "the",
      "regulating",
      "body",
      "of",
      "the",
      "Government",
      "of",
      "the",
      "Republic",
      "of",
      "the",
      "Philippines",
      "for",
      "civil",
      "aviation",
      ",",
      "suspended",
      "Zest",
      "Air",
      "flights",
      "until",
      "further",
      "notice",
      "because",
      "of",
      "safety",
      "issues",
      "."
    ],
    [
      "Less",
      "than",
      "a",
      "year",
      "after",
      "AirAsia",
      "and",
      "Zest",
      "Air",
      "\u0027s",
      "strategic",
      "alliance",
      ",",
      "the",
      "airline",
      "has",
      "been",
      "rebranded",
      "as",
      "AirAsia",
      "Zest",
      "."
    ],
    [
      "The",
      "airline",
      "was",
      "merged",
      "into",
      "AirAsia",
      "Philippines",
      "in",
      "January",
      "2016",
      "."
    ]
  ]
}
```

Please submit the test set result to Codalab.
