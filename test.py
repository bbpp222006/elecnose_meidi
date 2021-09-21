import configparser, json
import numpy as np


a={"num_sig":8,
"re_name_dic" : {
    '空': ['empty',1],
    "茶": ["tea",2],
    "咖啡": ["coffee",4],
    "雪碧": ["spirit",1],
    "可乐": ["cola",3],
    "白醋": ["white_vinegar",1],
    "老抽": ["dark_soysauce",4],
    "生抽": ["soy_sauce",4],
    "酒": ["wine",1],
    "糖水": ["sugar",1],
}
}
#
# filename='config.json'
# with open(filename,'w', encoding='utf-8') as file_obj:
#     json.dump(a,file_obj,ensure_ascii=False)
#
a=[1,2,3]
b=np.repeat([a],4,axis=0)
print(b)