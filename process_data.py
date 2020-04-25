import re
import pandas as pd
from numpy import array
from sklearn.cluster import KMeans,DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

def change_to_hotpot(answer, item):
      returnlist = [0]*len(item)
      for ele in answer:
        if ele in item:
          returnlist[item.index(ele)]=1
      return returnlist

def process_data( attribute_file, answer_sets):
        with open(attribute_file, 'r') as file:
            question = file.read()
        itemlist = [int(s) for s in re.findall(r"arg[(]a(.*?)[)].", question)]
        itemlist.sort()
        with open(answer_sets, 'r') as file:
            answer = file.read()
        test = answer.split("Answer:")
        del test[0]
        indexlist = [int(s.split("\n",1)[0]) for s in test]

        transfered=[]
        arguments=[]
        for s in test:
          temp1=re.findall(r"^in(.*)", s, re.M)
          if len(temp1)!=0:
            temp2=[int(s) for s in re.findall(r'\d+', temp1[0])]
            one_answer=change_to_hotpot(temp2,itemlist)
            transfered.append(one_answer)
            arguments.append(temp2)
          else:
            arguments.append(temp1)
            transfered.append([])

        not_defeated1=[]
        for s in test:
          temp1=re.findall(r"defeated(.*)", s, re.M)
          if len(temp1)!=0:
            temp2=[int(s) for s in re.findall(r'\d+', temp1[0])]
            not_defeated1.append(temp2)
          else:
            not_defeated1.append(temp1)

        processed_data = pd.DataFrame({
            'id': indexlist,
            'in': transfered,
            'arg': arguments,
            'not_defeated': not_defeated1,


        })
        processed_data["groups"]=["complete"]*len(processed_data)
        stable_idx=[]
        for index, row in processed_data.iterrows():
          if len(row.not_defeated) == 0:
            stable_idx.append(index)
        processed_data.loc[stable_idx,"groups"]=["stable"]*len(stable_idx)
        processed_data.loc[~processed_data.index.isin(stable_idx),"groups"]=["prefer-"]*(len(processed_data)-len(stable_idx))
        processed_data=clustering_dbscan(processed_data)
        processed_data=dimensional_reduction(processed_data)
        processed_data= clustering_km(processed_data)
        frequency = []
        for attribute in itemlist:
            count = 0
            for indx, row in processed_data.iterrows():
                if attribute in row.arg:
                    count += 1
            frequency.append(count),
        rate = [x / len(processed_data) * 100 for x in frequency]
        bar_data = pd.DataFrame({
            # "index":itemlist,
            "attribute": [str(x)+"attribute" for x in itemlist],
            "frequency": frequency,
            "rate": rate
        })
        #correlation matrix
        all_occurence = pd.DataFrame(columns=[str(x) for x in itemlist])
        for idx, raw in processed_data.iterrows():
            occurence = [0] * len(itemlist)
            for argument in itemlist:
                if argument in raw.arg:
                    occurence[itemlist.index(argument)] = 1
            all_occurence.loc[idx] = occurence
        select_column = []
        for arg in all_occurence.columns:
            if set(all_occurence[arg]) == {1} or set(all_occurence[arg]) == {0}:
                continue
            select_column.append(arg)
        selected = all_occurence[select_column]
        temp = selected.astype(int)
        correlation_matrix = temp.corr()

        #find features:
        cluster_feature_db=find_feature_cluster(itemlist,processed_data,cluster_algorithm,"db")
        cluster_feature_km=find_feature_cluster(itemlist,processed_data,cluster_algorithm,"km")
        group_feature=find_feature_group(itemlist,processed_data)
        #
        #
        #
        return processed_data,bar_data,correlation_matrix,cluster_feature_db,cluster_feature_km,group_feature


def clustering_km( data, cluster_num=2):
    km = KMeans(n_clusters=cluster_num, precompute_distances='auto').fit_predict(list(data['in']))
    data['km_cluster_label'] = km
    return data

def clustering_dbscan( data, eps=1.7, minpoint=7):
    c = DBSCAN(eps=eps, min_samples=minpoint).fit_predict(list(data['in']))
    data['db_cluster_label'] = c
    return  data


def dimensional_reduction(data) -> pd.DataFrame:
    result2 = TSNE(n_components=2).fit_transform(list(data['in'])).T
    data['tsne_position_x'] = result2[0]
    data['tsne_position_y'] = result2[1]

    svd = TruncatedSVD(n_components=2, n_iter=7)

    result = svd.fit_transform(list(data['in'])).T

    data['svd_position_x'] = result[0]
    data['svd_position_y'] = result[1]
    return data

def find_feature_cluster(itemlist, data, cluster_algorithm):  #clustered data
    cluster_label=cluster_algorithm+"_cluster_label"
    clusters = list(set(data[cluster_label]))
    characters = []
    for cluster in clusters:
        current_cluster = data[data[cluster_label] == cluster]
        all_lists = []
        for indx, row in current_cluster.iterrows():
            content = row['arg']
            all_lists.append(content)
        common_links = set(all_lists[0]).intersection(*all_lists[1:])
        characters.append(common_links)

    commen_arg = characters[0].intersection(*characters[1:])
    differences = [x - commen_arg for x in characters]
    for element in differences:
        flag = 1
        index = differences.index(element)
        current_cluster = clusters[index]
        other_cluster_data = data[data[cluster_label] != current_cluster]
        for idx, raw in other_cluster_data.iterrows():
            if set(element).issubset(set(raw.arg)):
                flag = 0
                print("{} is not character for cluster {}".format(element, current_cluster))
                differences[index]=[]
                break
        if flag:
            print("{} is character for cluster {}".format(element, current_cluster))
    sum_diff = pd.DataFrame({
        "cluster": clusters,
        "attribute_combination_feature": ["(" + ", ".join(str(x) for x in a) + ")" for a in differences],
    })
    return sum_diff

def find_feature_group(itemlist,data):
    record_character = dict({
        "attribute": [],
        "groups": [],
    })
    for attribute in itemlist:
        count = []
        for idx, raw in data.iterrows():
            if attribute in raw.arg:
                count.append(raw.groups)
        if len(set(count)) == 1:
            record_character["attribute"].append(attribute)
            record_character["groups"].append(count[0])
    if len(set(record_character["groups"]))==1:
        return  pd.DataFrame({})
    stable = []
    prefer = []
    temp=pd.DataFrame(record_character)
    for idx, raw in temp.iterrows():
        if raw.groups == "stable":
            stable.append(raw.attribute)
        else:
            prefer.append(raw.attribute)
    group_character = pd.DataFrame({
        "groups": ["stable", "prefer"],
        "attribute_feature": [", ".join(str(x) for x in stable), ', '.join(str(x) for x in prefer)]
    })
    return group_character