import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def sparsFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature_name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    : return
    """
    return {'feat': feat}


def create_cretio_data(embed_dim=4, test_size=0.1):
    # import data
    data_df = pd.read_csv('./train.txt',sep='\t',nrows=1000000) #100000
    # input("read")
    # data_df = data_df.sample(frac=0.001)
    # input("sample")
    data_df.columns=['Label','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13',
                    'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
                    'C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26']

    data_df_0 = data_df[data_df.Label == 0]
    data_df_1 = data_df[data_df.Label == 1]
    del data_df
    num = len(data_df_1)
    data_df_0 = data_df_0.sample(n=num)
    data_df = pd.concat([data_df_0,data_df_1])
    data_df.sample(frac=1).reset_index(drop=True)

    # 进行数据合并
    label = data_df['Label']
    del data_df['Label']

    print(data_df.columns)
    # 特征分开类别
    sparse_feas = [col for col in data_df.columns if col[0] == 'C']
    dense_feas = [col for col in data_df.columns if col[0] == 'I']

    # 填充缺失值
    data_df[sparse_feas] = data_df[sparse_feas].fillna('-1')
    data_df[dense_feas] = data_df[dense_feas].fillna(0)

    # 把特征列保存成字典, 方便类别特征的处理工作
    feature_columns = [[denseFeature(feat) for feat in dense_feas]] + [
        [sparsFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim) for feat in sparse_feas]]
    np.save('preprocessed_data/fea_col.npy', feature_columns)

    # 数据预处理
    # 进行编码  类别特征编码
    for feat in sparse_feas:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # 数值特征归一化
    mms = MinMaxScaler()
    data_df[dense_feas] = mms.fit_transform(data_df[dense_feas])

    data_df['Label'] = label

    # 划分验证集
    train_set, val_set = train_test_split(data_df, test_size=test_size, random_state=2022)

    # 保存文件
    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)

    train_set.to_csv('preprocessed_data/train_set.csv', index=0)
    val_set.to_csv('preprocessed_data/val_set.csv', index=0)

create_cretio_data()