import copy
import json
import math
import os
from collections import Counter
import xgboost as xgb
from itertools import chain
import datetime

import gc
from multiprocessing import Pool

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc


def drop_duplicate_record(record):
    if len(record) > 0:
        invite_answers = list(map(lambda x: x[0], filter(lambda x: x[2] == "1", record)))
        result = list(filter(lambda x: (x[0] in invite_answers and x[2] == "0") is False, record))
    else:
        result = []
    return result


def recover_dict_from_str(x):
    if x == "{}":
        return {}
    x = "\": ".join(x.split(": "))
    x = ", \"".join(x.split(", "))
    x = x.replace("{", "{\"")
    x = json.loads(x)
    y = copy.deepcopy(x)
    for key in y:
        del x[key]
        x[int(key)] = y[key]
    return x


def process_record(row):
    previous_records = []
    question = row["question"]
    # 删除本次回答之前的回答
    # TODO：只删除自己？后面保留？
    for record in row["answer_list"]:
        if record[0] != question:
            previous_records.append(record)
        else:
            break
    current_day, current_hour = extract_day_and_hour(row["time"])
    invite_records = list(filter(lambda x: x[2] == "1", previous_records))
    total_answer_days = list(map(lambda x: extract_day_and_hour(x[1])[0], previous_records))
    total_answer_hours = list(map(lambda x: extract_day_and_hour(x[1])[1], previous_records))
    invite_answer_days = list(map(lambda x: extract_day_and_hour(x[1])[0], invite_records))
    invite_answer_hours = list(map(lambda x: extract_day_and_hour(x[1])[1], invite_records))
    # 回答时间间隔最大值、最小值、趋势
    answer_days_diff = np.diff(total_answer_days).tolist() if len(total_answer_days) > 1 else []
    max_days_diff = max(answer_days_diff) if len(answer_days_diff) > 0 else np.nan
    min_days_diff = min(answer_days_diff) if len(answer_days_diff) > 0 else np.nan
    latest_num = math.ceil(len(previous_records) / 3)
    answer_trend = np.mean(answer_days_diff[-latest_num:]) / np.mean(answer_days_diff) if len(answer_days_diff) > 1 else 1
    # 距离最后一次回答、受邀时间间隔
    last_answer_day_gap = current_day - total_answer_days[-1] if len(total_answer_days) > 0 else 67
    last_invite_day_gap = current_day - invite_answer_days[-1] if len(invite_answer_days) > 0 else 37
    # 与所有回答/受邀 hour 的差最小值，与所有回答/受邀 hour 平均值的差
    mean_answer_hour = np.mean(total_answer_hours) if len(total_answer_hours) > 0 else np.nan
    mean_invite_hour = np.mean(invite_answer_hours) if len(invite_answer_hours) > 0 else np.nan
    answer_mean_hour_gap = abs(current_hour - mean_answer_hour) if not np.isnan(mean_answer_hour) else np.nan
    invite_mean_hour_gap = abs(current_hour - mean_invite_hour) if not np.isnan(mean_invite_hour) else np.nan
    answer_min_hour_gap = min(list(map(lambda x: abs(x - current_hour), total_answer_hours))) if len(total_answer_hours) > 0 else np.nan
    invite_min_hour_gap = min(list(map(lambda x: abs(x - current_hour), invite_answer_hours))) if len(invite_answer_hours) > 0 else np.nan
    # 回答/受邀各个区间的数量、总数量、受邀回答比
    answer_num_1, answer_num_2, answer_num_3, answer_num_4, answer_num_5, answer_num_6 = get_detail_answer_num(total_answer_days)
    invite_num_1, invite_num_2, invite_num_3, invite_num_4, invite_num_5, invite_num_6 = get_detail_invite_num(invite_answer_days)
    record_num = len(previous_records)
    invite_num = len(invite_records)
    invite_ratio = invite_num / record_num if record_num > 0 else np.nan
    # 回答在一周内分布，该邀请所对应的数目
    week_num_list = get_week_answer_num(total_answer_days)
    week_num_1, week_num_2, week_num_3, week_num_4, week_num_5, week_num_6, week_num_7 = week_num_list
    week_num = week_num_list[int(current_day % 7)]
    # 邀请对应的星期和时间
    current_week = str(current_day % 7)
    current_hour = str(current_hour)
    current_week_hour = current_week + "-" + current_hour
    # 最后一次是回答，受邀，还是无
    if len(previous_records) == 0:
        last_type = "-1"
    elif previous_records[-1][2] == "0":
        last_type = "0"
    else:
        last_type = "1"
    return (previous_records, max_days_diff, min_days_diff, answer_trend, last_answer_day_gap, last_invite_day_gap,
            answer_mean_hour_gap, invite_mean_hour_gap, answer_min_hour_gap, invite_min_hour_gap, answer_num_1,
            answer_num_2, answer_num_3, answer_num_4, answer_num_5, answer_num_6, invite_num_1, invite_num_2,
            invite_num_3, invite_num_4, invite_num_5, invite_num_6, record_num, invite_num, invite_ratio, week_num_1,
            week_num_2, week_num_3, week_num_4, week_num_5, week_num_6, week_num_7, week_num, current_week,
            current_hour, current_week_hour, last_type)


def get_embedding():
    embedding = []
    init_embedding = np.random.uniform(-1, 1, 64).tolist()
    embedding.append(init_embedding)
    with open(data_dir + "topic_vectors_64d.txt", "r") as f:
        for i, line in enumerate(f):
            line = line.strip().split("\t")
            key = line[0].replace("T", "")
            assert int(key) == i + 1
            embedding.append(list(map(float, line[1].split(" "))))
    return np.array(embedding)


def get_answer_topics_and_weight(record):
    topic_list = []
    weight_list = []
    topics = list(map(lambda x: question_topic_dict[x[0]], record))
    topic_count_dict = dict(Counter(list(chain(*topics))))
    for topic, weight in topic_count_dict.items():
        topic_list.append(topic)
        weight_list.append(weight)
    return topic_list, weight_list


def get_interest_topics_and_weight(topic_dict):
    topic_list = []
    weight_list = []
    for topic, weight in topic_dict.items():
        topic_list.append(topic)
        weight_list.append(weight)
    return topic_list, weight_list


def cal_cos_dist(list_a, list_b):
    if len(list_a) == 0 or len(list_b) == 0:
        return [], []
    a = np.array(list(map(lambda x: embedding[x], list_a)))
    b = np.array(list(map(lambda x: embedding[x], list_b)))
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    ab_norm = np.matmul(a_norm[:, np.newaxis], b_norm[np.newaxis, :])
    ab_dot = np.matmul(a, np.swapaxes(b, 0, 1))
    cos_matrix = np.true_divide(ab_dot, ab_norm)
    a_b_max_pool = np.max(cos_matrix, axis=1).tolist()
    b_a_max_pool = np.max(cos_matrix, axis=0).tolist()
    return a_b_max_pool, b_a_max_pool


def extract_day_and_hour(time_str):
    day, hour = list(map(lambda x: int(x), time_str.replace("D", "").replace("H", "").split("-")))
    return day, hour


def get_detail_answer_num(total_answer_days):
    answer_num_1, answer_num_2, answer_num_3, answer_num_4, answer_num_5, answer_num_6 = 0, 0, 0, 0, 0, 0
    for day in total_answer_days:
        relative_day = day - 3807
        if 0 <= relative_day < 10:
            answer_num_1 += 1
        elif 10 <= relative_day < 20:
            answer_num_2 += 1
        elif 20 <= relative_day < 30:
            answer_num_3 += 1
        elif 30 <= relative_day < 40:
            answer_num_4 += 1
        elif 40 <= relative_day < 50:
            answer_num_5 += 1
        else:
            answer_num_6 += 1
    return answer_num_1, answer_num_2, answer_num_3, answer_num_4, answer_num_5, answer_num_6


def get_detail_invite_num(invite_answer_days):
    invite_num_1, invite_num_2, invite_num_3, invite_num_4, invite_num_5, invite_num_6 = 0, 0, 0, 0, 0, 0
    for day in invite_answer_days:
        relative_day = day - 3838
        if 0 <= relative_day < 5:
            invite_num_1 += 1
        elif 5 <= relative_day < 10:
            invite_num_2 += 1
        elif 10 <= relative_day < 15:
            invite_num_3 += 1
        elif 15 <= relative_day < 20:
            invite_num_4 += 1
        elif 20 <= relative_day < 25:
            invite_num_5 += 1
        else:
            invite_num_6 += 1
    return invite_num_1, invite_num_2, invite_num_3, invite_num_4, invite_num_5, invite_num_6


def get_week_answer_num(total_answer_days):
    answer_num_1, answer_num_2, answer_num_3, answer_num_4, answer_num_5, answer_num_6, answer_num_7 = 0, 0, 0, 0, 0, 0, 0
    for day in total_answer_days:
        week = day % 7
        if week == 0:
            answer_num_1 += 1
        elif week == 1:
            answer_num_2 += 1
        elif week == 2:
            answer_num_3 += 1
        elif week == 3:
            answer_num_4 += 1
        elif week == 4:
            answer_num_5 += 1
        elif week == 5:
            answer_num_6 += 1
        else:
            answer_num_7 += 1
    return answer_num_1, answer_num_2, answer_num_3, answer_num_4, answer_num_5, answer_num_6, answer_num_7


def cal_topic_weighted_score(row):
    answer_weighted_sum = np.sum(np.array(row["answer_topics_weight"]) * np.array(row["answer_to_question_score"])) if len(row["answer_topics_weight"]) > 0 else 0
    answer_weighted_mean = answer_weighted_sum / np.sum(row["answer_topics_weight"]) if answer_weighted_sum != 0 else 0
    interest_weighted_sum = np.sum(np.array(row["interest_topics_weight"]) * np.array(row["interest_to_question_score"])) if len(row["interest_topics_weight"]) > 0 else 0
    interest_weighted_mean = interest_weighted_sum / np.sum(row["interest_topics_weight"]) if interest_weighted_sum != 0 else 0
    return answer_weighted_sum, answer_weighted_mean, interest_weighted_sum, interest_weighted_mean


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
    # 处理逗号分隔的 words、topics，W 和 T 要跳过，转成 int
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))


def parse_list_2(d):
    # single words，SW 开头要跳过，转成 int
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[2:]), str(d).split(',')))


def parse_map(d):
    # 处理逗号、分号构成的 topcis:interest 字典
    if d == '-1':
        return {}
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))


class Timer:
    def __init__(self, s="开始"):
        self.time_list = [datetime.datetime.now()]
        self.time_dict = {s: 0}

    def print_time(self, s, last_s=None):
        self.time_dict[s] = len(self.time_list)
        current_time = datetime.datetime.now()
        last_index = self.time_dict[last_s] if last_s else -1
        print("%s:" %s, current_time - self.time_list[last_index])
        self.time_list.append(current_time)


def process_invite(index, invite_df):
    # 去掉 answer 中重复的问题（即使 answer 又是 invite）
    invite_df["answer_list"] = invite_df["answer_list"].apply(lambda x: x if isinstance(x, list) else [])
    invite_df["answer_list"] = invite_df["answer_list"].apply(drop_duplicate_record)
    # 处理回答记录
    (invite_df["previous_answer"], invite_df["max_days_diff"], invite_df["min_days_diff"], invite_df["answer_trend"],
     invite_df["last_answer_day_gap"], invite_df["last_invite_day_gap"], invite_df["answer_mean_hour_gap"],
     invite_df["invite_mean_hour_gap"], invite_df["answer_min_hour_gap"], invite_df["invite_min_hour_gap"],
     invite_df["answer_num_1"], invite_df["answer_num_2"], invite_df["answer_num_3"], invite_df["answer_num_4"],
     invite_df["answer_num_5"], invite_df["answer_num_6"], invite_df["invite_num_1"], invite_df["invite_num_2"],
     invite_df["invite_num_3"], invite_df["invite_num_4"], invite_df["invite_num_5"], invite_df["invite_num_6"],
     invite_df["record_num"], invite_df["invite_num"], invite_df["invite_ratio"], invite_df["week_num_1"],
     invite_df["week_num_2"], invite_df["week_num_3"], invite_df["week_num_4"], invite_df["week_num_5"],
     invite_df["week_num_6"], invite_df["week_num_7"], invite_df["week_num"], invite_df["current_week"],
     invite_df["current_hour"], invite_df["current_week_hour"], invite_df["last_type"]
     ) = zip(*invite_df.apply(process_record, axis=1))
    # 得到回答和感兴趣 topics 的 topic_list 和 weight_list
    invite_df["answer_topics"], invite_df["answer_topics_weight"] = zip(*invite_df["previous_answer"].apply(get_answer_topics_and_weight))
    invite_df["interest_topics"], invite_df["interest_topics_weight"] = zip(*invite_df["interest_topics"].apply(get_interest_topics_and_weight))
    # 将 回答、感兴趣、关注、问题的 topic_list 映射成向量，计算点乘得分
    invite_df["answer_to_question_score"], invite_df["question_to_answer_score"] = zip(*invite_df.apply(lambda row: cal_cos_dist(row["answer_topics"], row["topics"]), axis=1))
    invite_df["interest_to_question_score"], invite_df["question_to_interest_score"] = zip(*invite_df.apply(lambda row: cal_cos_dist(row["interest_topics"], row["topics"]), axis=1))
    invite_df["subscribe_to_question_score"], invite_df["question_to_subscribe_score"] = zip(*invite_df.apply(lambda row: cal_cos_dist(row["subscribe_topics"], row["topics"]), axis=1))
    # 回答、感兴趣的 topics 对问题 topics 加权分数
    (invite_df["answer_weighted_sum"], invite_df["answer_weighted_mean"], invite_df["interest_weighted_sum"],
     invite_df["interest_weighted_mean"]) = zip(*invite_df.apply(cal_topic_weighted_score, axis=1))
    # 问题 topics 对回答、感兴趣、关注 topics 的最大值、最小值、均值、峰度、高分个数、高分比例
    invite_df["question_to_answer_score_max"] = invite_df["question_to_answer_score"].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
    invite_df["question_to_answer_score_min"] = invite_df["question_to_answer_score"].apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
    invite_df["question_to_answer_score_mean"] = invite_df["question_to_answer_score"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    invite_df["question_to_answer_score_kurt"] = invite_df["question_to_answer_score"].apply(lambda x: stats.kurtosis(x) if len(x) > 0 else np.nan)
    invite_df["question_to_answer_high_score_num"] = invite_df["question_to_answer_score"].apply(lambda x: np.sum(np.array(x) > 0.8) if len(x) > 0 else 0)
    invite_df["question_to_answer_high_score_ratio"] = invite_df["question_to_answer_score"].apply(lambda x: np.sum(np.array(x) > 0.8) / len(x) if len(x) > 0 else 0)
    invite_df["question_to_interest_score_max"] = invite_df["question_to_interest_score"].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
    invite_df["question_to_interest_score_min"] = invite_df["question_to_interest_score"].apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
    invite_df["question_to_interest_score_mean"] = invite_df["question_to_interest_score"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    invite_df["question_to_interest_score_kurt"] = invite_df["question_to_interest_score"].apply(lambda x: stats.kurtosis(x) if len(x) > 0 else np.nan)
    invite_df["question_to_interest_score_num"] = invite_df["question_to_interest_score"].apply(lambda x: np.sum(np.array(x) > 0.8) if len(x) > 0 else 0)
    invite_df["question_to_interest_score_ratio"] = invite_df["question_to_interest_score"].apply(lambda x: np.sum(np.array(x) > 0.8) / len(x) if len(x) > 0 else 0)
    invite_df["question_to_subscribe_score_max"] = invite_df["question_to_subscribe_score"].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
    invite_df["question_to_subscribe_score_min"] = invite_df["question_to_subscribe_score"].apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
    invite_df["question_to_subscribe_score_mean"] = invite_df["question_to_subscribe_score"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    invite_df["question_to_subscribe_score_kurt"] = invite_df["question_to_subscribe_score"].apply(lambda x: stats.kurtosis(x) if len(x) > 0 else np.nan)
    invite_df["question_to_subscribe_score_num"] = invite_df["question_to_subscribe_score"].apply(lambda x: np.sum(np.array(x) > 0.8) if len(x) > 0 else 0)
    invite_df["question_to_subscribe_score_ratio"] = invite_df["question_to_subscribe_score"].apply(lambda x: np.sum(np.array(x) > 0.8) / len(x) if len(x) > 0 else 0)
    # TODO: 对于importance高的score特征，可以做细分，问题topic score，多少到了0.95，0.8，0.5
    # 问题创建时间与现在时间差值
    invite_df["question_duration"] = invite_df.apply(lambda row: extract_day_and_hour(row["time"])[0] - extract_day_and_hour(row["question_time"])[0], axis=1)
    # Drop 不进模型的字段
    invite_df = invite_df.drop(["question", "member", "time", "original_time", "creation_keywrods", "creation_level",
                                "creation_popularity", "register_type", "register_platform", "subscribe_topics",
                                "interest_topics", "question_time", "title_single_words", "title_words",
                                "content_single_words", "content_words", "topics", "answer_list", "previous_answer",
                                "answer_topics", "answer_topics_weight", "interest_topics_weight",
                                "answer_to_question_score", "question_to_answer_score", "interest_to_question_score",
                                "question_to_interest_score", "subscribe_to_question_score",
                                "question_to_subscribe_score"], axis=1)
    return index, invite_df


def split_df(df, n):
    chunk_size = int(np.ceil(len(df) / n))
    return [df[i*chunk_size:(i+1)*chunk_size] for i in range(n)]


def gc_mp(pool, apply_results, result_list, chunk_list):
    for ar in apply_results:
        del ar
    for r in result_list:
        del r
    for cl in chunk_list:
        del cl
    del pool
    del result_list
    del apply_results
    del chunk_list
    gc.collect()


if __name__ == "__main__":
    if os.path.exists("/Volumes/hedongfeng/数据集/专家发现/data_set_0926/"):
        data_dir = "/Volumes/hedongfeng/数据集/专家发现/data_set_0926/"
    else:
        data_dir = "/root/ctr2/data_set_0926/"
    # answer_info_0926、invite_info_evaluate_1_0926、invite_info_0926、member_info_0926、question_info_0926
    # single_word_vectors_64d、topic_vectors_64d、word_vectors_64d
    timer = Timer("开始")
    embedding = get_embedding()
    # 正负例比例、
    answer_df = pd.read_csv(data_dir + "answer_info_0926.txt",
                            header=None, sep='\t',
                            # nrows=100
                            )
    invite_df = pd.read_csv(data_dir + "invite_info_0926.txt",
                            header=None, sep='\t',
                            # nrows=100
                            )
    invite_test_df = pd.read_csv(data_dir + "invite_info_evaluate_1_0926.txt",
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
    invite_test_df.columns = ["question", "member", "time"]
    question_df.columns = ["question", "question_time", "title_single_words", "title_words", "content_single_words",
                           "content_words", "topics"]
    member_df.columns = ["member", "gender", "creation_keywrods", "creation_level", "creation_popularity",
                         "register_type", "register_platform", "frequency", "bi_feat1", "bi_feat2", "bi_feat3",
                         "bi_feat4", "bi_feat5", "mul_feat1", "mul_feat2", "mul_feat3", "mul_feat4", "mul_feat5",
                         "salt_score", "subscribe_topics", "interest_topics"]
    timer.print_time("加载数据")
    """
    处理 answer_df、invite_df
    """
    answer_df = answer_df[["question", "member", "time"]]
    # 修正时间的位数
    answer_df["time"] = answer_df["time"].apply(fix_time)
    invite_df["original_time"] = invite_df["time"]
    invite_test_df["original_time"] = invite_test_df["time"]
    invite_df["time"] = invite_df["time"].apply(fix_time)
    invite_test_df["time"] = invite_test_df["time"].apply(fix_time)
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
    # 制作 question_topic_dict
    question_topic_df = question_df[["question", "topics"]]
    question_topic_dict = {row[0]: row[1] for index, row in question_topic_df.iterrows()}
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
    将 member_df、question_df 和 member_record 合并进 invite_df、invite_test_df
    """
    invite_df = pd.merge(invite_df, member_df, how='left', on='member')
    invite_df = pd.merge(invite_df, question_df, how='left', on='question')
    invite_df = pd.merge(invite_df, member_record, how='left', on='member')
    train_num = len(invite_df)
    invite_test_df = pd.merge(invite_test_df, member_df, how='left', on='member')
    invite_test_df = pd.merge(invite_test_df, question_df, how='left', on='question')
    invite_test_df = pd.merge(invite_test_df, member_record, how='left', on='member')
    invite_df = pd.concat([invite_df, invite_test_df], axis=0, sort=True)
    result_append = invite_df[["question", "member", "original_time"]][train_num:]
    # 删掉不用的 dataframe
    del member_df, question_df, member_record, answer_df, invite_success_df
    gc.collect()
    """
    处理 invite_df
    """
    chunk_list = split_df(invite_df, 160)
    del invite_df
    gc.collect()
    pool = Pool(processes=16)
    apply_results = []
    for index, chunk_df in enumerate(chunk_list):
        apply_results.append(pool.apply_async(process_invite, (index, chunk_df,)))
    pool.close()
    pool.join()
    result_list = []
    for res in apply_results:
        result = res.get()
        result_list.append(result)
    result_list.sort(key=lambda x: x[0])
    result_df_list = list(map(lambda x: x[1], result_list))
    invite_df = pd.concat(result_df_list)
    gc_mp(pool, apply_results, result_list, chunk_list)
    # 类别特征转换
    # TODO: '用户id','问题id' 要保留？
    class_feat = ['gender', 'frequency', 'bi_feat1', 'bi_feat2', 'bi_feat3', 'bi_feat4', 'bi_feat5', 'mul_feat1',
                  'mul_feat2', 'mul_feat3', 'mul_feat4', 'mul_feat5', 'current_week', 'current_hour',
                  'current_week_hour', 'last_type']
    encoder = LabelEncoder()
    for feat in class_feat:
        encoder.fit(invite_df[feat])
        invite_df[feat] = encoder.transform(invite_df[feat])
    timer.print_time("处理数据")
    """
    训练
    """
    param = {"n_estimators": 2000,
             'learning_rate ': 0.01,
             'silent': 1,
             'objective': 'binary:logistic',
             "eval_metric": "auc",
             "subsample": 0.8,
             "min_child_weight": 5,
             "n_jobs": -1,
             }
    invite_df.to_csv('train_df.txt', index=False, header=False, sep='\t')
    print(invite_df.columns)
    print(train_num)
    result_append.to_csv('result_append.txt', index=False, header=False, sep='\t')

    y_train = invite_df[:train_num]["label"].values
    x_train = invite_df[:train_num].drop(["label"], axis=1).values
    x_test = invite_df[train_num:].drop(["label"], axis=1).values


    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)

    cv_res = xgb.cv(param, dtrain, num_boost_round=2000, early_stopping_rounds=30, nfold=5, metrics='auc',
                    show_stdv=True)
    print(cv_res)
    bst = xgb.train(param, dtrain, num_boost_round=cv_res.shape[0])

    timer.print_time("训练时间")

    y_pred = bst.predict(dtest)

    result_append['proba'] = y_pred
    result_append.to_csv('result.txt', index=False, header=False, sep='\t')
