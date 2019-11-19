import pandas as pd
from collections import Counter
import math
import json
import numpy as np
import datetime
import pickle
import os


def add_to_dict(item, my_dict):
    if item in my_dict:
        my_dict[item] += 1
    else:
        my_dict[item] = 1


def convert_date(date):
    year, month, day = map(lambda x: int(x), date.split("-"))
    return datetime.date(year, month, day)


def filter_and_sort(*dicts):
    result_list = []
    for tmp_dict in dicts:
        sorted_list = sorted(tmp_dict.items(), key=lambda x: x[1], reverse=True)
        filtered_list = list(filter(lambda x: x[0] != '', sorted_list))
        result_list.append(filtered_list)
    return result_list


def filter_and_bucketing(tmp_list, bucket_num):
    filtered_list = list(filter(lambda x: x != '', tmp_list))
    sorted_list = sorted(filtered_list)
    avg_bucket_sample_num = len(filtered_list) / bucket_num
    bucket_list = [sorted_list[math.floor(avg_bucket_sample_num * i)] for i in range(1, bucket_num)]
    return bucket_list


def filter_and_indexing(sorted_list, threshold=0):
    sorted_list = list(filter(lambda x: x[1] > threshold and x[0] != "未知", sorted_list))
    index = 1
    index_dict = {}
    for key, cnt in sorted_list:
        index_dict[key] = index
        index += 1
    return index_dict


def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def fix_time(x):
    x = x.split("-")
    hour = x[1][1:].zfill(2)
    return "%s-%s" % (x[0], hour)


def parse_list_1(d):
    # 处理逗号分隔的 words
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))


def parse_list_2(d):
    # 处理逗号分隔的 single words 和 topics
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[2:]), str(d).split(',')))


def parse_map(d):
    # 处理逗号、分号构成的 topcis:interest 字典
    if d == '-1':
        return {}
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))


if __name__ == "__main__":
    if os.path.exists("/Volumes/hedongfeng/数据集/专家发现/data_set_0926/"):
        data_dir = "/Volumes/hedongfeng/数据集/专家发现/data_set_0926/"
    else:
        data_dir = "/root/ctr2/data_set_0926/"
    # answer_info_0926、invite_info_evaluate_1_0926、invite_info_0926、member_info_0926、question_info_0926
    # single_word_vectors_64d、topic_vectors_64d、word_vectors_64d
    # 正负例比例、
    answer_df = pd.read_csv(data_dir + "answer_info_0926.txt",
                            header=None, sep='\t',
                            # nrows=100
                            )
    invite_df = pd.read_csv(data_dir + "invite_info_0926.txt",
                            header=None, sep='\t',
                            # nrows=100
                            )
    member_df = pd.read_csv(data_dir + "member_info_0926.txt",
                            header=None, sep='\t',
                            # nrows=100
                            )
    question_df = pd.read_csv(data_dir + "question_info_0926.txt",
                              header=None, sep='\t',
                              # nrows=100
                              )
    answer_df.columns = ["answer", "question", "member", "time", "answer_single_words", "answer_words", "great_flag",
                         "rec_flag", "round_flag", "has_pic", "has_video", "word_cnt", "upvote_cnt", "upvote_cancel_cnt",
                         "comment_cnt", "collect_cnt", "thank_cnt", "report_cnt", "helpless_cnt", "downvote_cnt"]
    invite_df.columns = ["question", "member", "time", "label"]
    question_df.columns = ["question", "time", "title_single_words", "title_words", "content_single_words",
                           "content_words", "topics"]
    member_df.columns = ["member", "gender", "creation_keywrods", "creation_level", "creation_popularity",
                         "register_type", "register_platform", "frequency", "bi_feat1", "bi_feat2", "bi_feat3",
                         "bi_feat4", "bi_feat5", "mul_feat1", "mul_feat2", "mul_feat3", "mul_feat4", "mul_feat5",
                         "salt_score", "subscribe_topics", "interest_topics"]
    """
    处理 answer_df、invite_df
    """
    answer_df = answer_df[["question", "member", "time"]]
    # 修正时间的位数
    answer_df["time"] = answer_df["time"].apply(fix_time)
    invite_df["time"] = invite_df["time"].apply(fix_time)
    # 添加类型，回答是0，邀请是1
    invite_success_df = invite_df[invite_df["label"] == 1]
    invite_success_df = invite_success_df.drop(["label"], axis=1)
    answer_df["type"] = "0"
    invite_success_df["type"] = "1"
    # 将问题、时间、类型合并成 list
    answer_df["info"] = answer_df["question"].str.cat(answer_df[["time", "type"]], sep="|")
    invite_success_df["info"] = invite_success_df["question"].str.cat(invite_success_df[["time", "type"]], sep="|")
    answer_df["info"] = answer_df["info"].apply(lambda x: x.split("|"))
    invite_success_df["info"] = invite_success_df["info"].apply(lambda x: x.split("|"))
    # 每个用户的回答/受邀列表
    member_answers = answer_df.groupby("member")["info"].apply(list).reset_index()
    member_invites = invite_success_df.groupby("member")["info"].apply(list).reset_index()
    member_record = pd.merge(member_answers, member_invites, how="outer", on=["member"])
    member_record["info_x"] = member_record["info_x"].apply(lambda x: x if isinstance(x, list) else [])
    member_record["info_y"] = member_record["info_y"].apply(lambda x: x if isinstance(x, list) else [])
    member_record["answer_list"] = member_record["info_x"] + member_record["info_y"]
    member_record = member_record.drop(["info_x", "info_y"], axis=1)
    # 问题记录按时间排序
    member_record["answer_list"] = member_record["answer_list"].apply(lambda x: sorted(x, key=lambda y: y[1]))

    """
    处理 question_df
    """
    # 标题、内容 str 解析成 list
    question_df['title_single_words'] = question_df['title_single_words'].apply(parse_list_2)
    question_df['title_words'] = question_df['title_words'].apply(parse_list_1)
    question_df['content_single_words'] = question_df['content_single_words'].apply(parse_list_2)
    question_df['content_words'] = question_df['content_words'].apply(parse_list_1)
    question_df['topics'] = question_df['topics'].apply(parse_list_1)
    # title、desc 词计数，topic 计数
    question_df['num_title_words'] = question_df['title_single_words'].apply(len)
    question_df['num_desc_words'] = question_df['content_single_words'].apply(len)
    question_df['num_topics'] = question_df['topics'].apply(len)

    """
    处理 member_df
    """
    # 将 topic 解析成 list 和 dict
    member_df['subscribe_topics'] = member_df['subscribe_topics'].apply(parse_list_1)
    member_df['interest_topics'] = member_df['interest_topics'].apply(parse_map)
    # 用户关注和感兴趣的 topic 数
    member_df['num_subscribe_topics'] = member_df['subscribe_topics'].apply(len)
    member_df['num_interest_topics'] = member_df['interest_topics'].apply(len)  # 人工计算，上限为10

    """
    将 member_df 和 question_df 合并进 invite_df
    """
    invite_df = pd.merge(invite_df, member_df, how='left', on='member')
    invite_df = pd.merge(invite_df, question_df, how='left', on='question')
    invite_df.to_csv("invite_df.csv", index=False, sep='\t')
    """
    处理 question 和 member 交互
    """
    # df.apply(lambda row: list(set(row['topic_interest'].keys()) & set(row['topic'])), axis=1)

