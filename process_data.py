import re
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from os import listdir,unlink,walk
from os.path import isfile, join
from zipfile import ZipFile
from pathlib import Path
import time
import itertools

def clean_folder(folder_path):
    if len(listdir(folder_path))!=0:
        removed=[]
        for the_file in listdir(folder_path):
            file_path = join(folder_path, the_file)
            try:
                if isfile(file_path):
                    unlink(file_path)
                    removed.append(the_file)
            except Exception as e:
                print(e)
        return removed
    else:
        return []


# def change_to_hotpot(answer, item):
#
#       returnlist = [0]*len(item)
#       for ele in answer:
#         if ele in item:
#           returnlist[item.index(ele)]=1
#       return returnlist


def f_comma(my_str, group, char=','):
        my_str = str(my_str)
        return char.join(my_str[i:i+group] for i in range(0, len(my_str), group))


def process_data(dir, arguments_file, answer_sets, eps, minpts, n_cluster):
        start = time.process_time()
        with open(arguments_file, 'r') as file:
            question = file.read()
        itemlist = [int(s) for s in re.findall(r"arg[(]a(.*?)[)].", question)]
        itemlist.sort()
        with open(answer_sets, 'r') as file:
            answer = file.read()
        if 'EE-PR' in answer_sets:
            semantic="prefer"
        elif "EE-STG" in answer_sets:
            semantic="stage"
        test = answer.split("Answer:")
        del test[0]
        #indexlist = [int(s.split("\n",1)[0]) for s in test]
        arg_len=len(test)
        indexlist = range(1,arg_len+1)
        transfered=[]
        arguments=[]


        for s in test:
          temp1=re.findall(r"^in(.*)", s, re.M)
          if temp1:
            temp2=[int(s) for s in re.findall(r'\d+', temp1[0])]
            bool_represent = np.in1d(itemlist, temp2) #boolean list representation
            one_answer=bool_represent.astype(int) #to int list
            transfered.append( one_answer)
            arguments.append(temp2)
          else:
            arguments.append([])
            transfered.append([])


        if semantic=="prefer":
            not_defeated1 = []
            for s in test:
                temp1 = re.findall(r"defeated(.*)", s, re.M)
                if len(temp1) != 0:
                    temp2 = [int(s) for s in re.findall(r'\d+', temp1[0])]
                    not_defeated1.append(temp2)
                else:
                    not_defeated1.append(0)
            processed_data = pd.DataFrame({
                'id': indexlist,
                'in': transfered,
                'arg': arguments,
                'not_defeated': not_defeated1,
            })
            processed_data["groups"] = np.where(processed_data["not_defeated"] == 0, "stable", "preferred")

            print("(prefer)generally process answer sets( read data, one hot, group label, ): ", time.process_time() - start)
            start2 = time.process_time()
            if eps != "" and eps != "Eps":
                processed_data = clustering_dbscan(processed_data, float(eps), int(minpts))
            else:
                processed_data = clustering_dbscan(processed_data)
            print("(prefer)dbscan clustering: ", time.process_time() - start2)
        else:
            not_defeated1=[]
            for s in test:
                temp1 = re.findall(r"nrge(.*)", s, re.M)
                if len(temp1) != 0:
                    temp2 = [int(s) for s in re.findall(r'\d+', temp1[0])]
                    not_defeated1.append(temp2)
                else:
                    not_defeated1.append(0)
            processed_data = pd.DataFrame({
                'id': indexlist,
                'in': transfered,
                'arg': arguments,
                'nrge': not_defeated1,
            })
            processed_data["groups"] = np.where(processed_data["nrge"] == 0, "stable", "stage")


            print("(stgae)generally process answer sets( read data, one hot, group label, ): ", time.process_time() - start)
            start2 = time.process_time()
            if eps != "" and eps != "Eps":
                processed_data = clustering_dbscan(processed_data, float(eps), int(minpts))
            else:
                processed_data = clustering_dbscan(processed_data)
            print("(stgae)dbscan clustering: ", time.process_time() - start2)

        processed_data = dimensional_reduction(processed_data)
        start3 = time.process_time()

        if n_cluster !="" and n_cluster !="Cluster Num":
            processed_data= clustering_km(processed_data,int(n_cluster))
        else:
            processed_data= clustering_km(processed_data)
        print("kmeans clustering: ", time.process_time() - start3)

        start4 = time.process_time()
        all_arguments = [item for sublist in arguments for item in sublist]
        frequency = np.array([])
        for argument in itemlist:
            #count = 0

            # for arg_list in arguments:
            #     if argument in arg_list:
            #         count += 1
            frequency=np.append(frequency,all_arguments.count(argument))
        rate=frequency / len(processed_data) *100
        bar_data = pd.DataFrame({
            # "index":itemlist,
            "argument": itemlist,#argument
            "frequency": frequency,
            "rate": rate
        })
        print("bar chart(argument frequency): ", time.process_time() - start4)
        #correlation matrix
        start5 = time.process_time()
        all_occurence=pd.DataFrame([x for x in processed_data['in']], columns=[str(x) for x in itemlist])
        to_drop = bar_data.loc[bar_data['rate'].isin([0, 100])].argument
        all_occurence.drop([str(x) for x in to_drop], axis='columns', inplace=True)

        temp = all_occurence.astype(int)
        correlation_matrix = temp.corr()
        print("create correlation matrix: ", time.process_time() - start5)

        #find features:
        start6 = time.process_time()
        common_all = set(arguments[0]).intersection(*arguments)
        cluster_feature_db=find_feature_cluster(common_all,processed_data,"db")
        cluster_feature_km=find_feature_cluster(common_all,processed_data,"km")
        group_feature=find_feature_cluster(common_all,processed_data,"groups")
        #
        #
        processed_data_dir=dir+semantic
        group_feature.to_pickle(processed_data_dir + "_group_feature.pkl")
        cluster_feature_db.to_pickle(processed_data_dir + "_db_cluster_feature.pkl")
        cluster_feature_km.to_pickle(processed_data_dir + "_km_cluster_feature.pkl")
        print("create feature report: ", time.process_time() - start6)
        processed_data.to_pickle(processed_data_dir + "_processed_data.pkl")
        bar_data["argument"]=[str(x)+"argument" for x in itemlist]
        bar_data.to_pickle(processed_data_dir + "_bar_data.pkl")
        correlation_matrix.to_pickle(processed_data_dir + "_correlation_matrix.pkl")

        # create a ZipFile object
        start7 = time.process_time()
        file_name=arguments_file.split("/")[-1]
        zipname = semantic+"_"+file_name.strip("apx") + "zip"
        path = Path(dir)
        zip_dir= str(path.parent) +"/processed_zip/"
        with ZipFile(zip_dir + zipname, 'w') as zipObj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in walk(dir):
                for filename in filenames:
                    if filename.find(semantic)==0:
                        # create complete filepath of file in directory
                        filePath = join(folderName, filename)
                        # Add file to zip
                        zipObj.write(filePath, arcname=filename)
        print("zip files: ", time.process_time() - start7)
        #return processed_data,bar_data,correlation_matrix,cluster_feature_db,cluster_feature_km,group_feature


def clustering_km( data, cluster_num=2):
    km = KMeans(n_clusters=cluster_num, precompute_distances='auto').fit_predict(list(data['in']))
    data['km_cluster_label'] = km
    return data

def clustering_dbscan( data, eps=1.7, minpoint=7):
    c = DBSCAN(eps=eps, min_samples=minpoint).fit_predict(list(data['in']))
    data['db_cluster_label'] = c
    return  data


def dimensional_reduction(data) -> pd.DataFrame:
    start1 = time.process_time()
    result2 = TSNE(n_components=2).fit_transform(list(data['in'])).T
    data['tsne_position_x'] = result2[0]
    data['tsne_position_y'] = result2[1]
    print("Tsne dimensional reduciton: ", time.process_time() - start1)
    svd = TruncatedSVD(n_components=2, n_iter=7)
    start2 = time.process_time()
    result = svd.fit_transform(list(data['in'])).T
    print("SVD dimensional reduciton: ", time.process_time() - start2)
    data['svd_position_x'] = result[0]
    data['svd_position_y'] = result[1]
    return data

def add_to(feature, combine):
  mask=[combine.issubset(x) for x in feature]
  if any(mask):
    super_f=list(itertools.compress(feature,mask))
    for x in super_f:
      feature.remove(x)

    feature.append(combine)

def find_feature_cluster(common_all, data, labels):  #clustered data
    if labels !="groups":
        labels=labels+"_cluster_label"


    clusters = data[labels].unique().tolist()
    if len(clusters)==1 or len(clusters)>50:
        return pd.DataFrame([])
    all_feature = []
    cluster_with_feature = clusters.copy()
    for cluster in clusters:
        feature = []
        current_cluster = data[data[labels] == cluster]
        all_lists = list(current_cluster.arg)
        common_links = set(all_lists[0]).intersection(*all_lists[1:])
        common_links = common_links - common_all

        #other_arguments = [item for sublist in other_cluster_arg for item in sublist]
        # mask = [a not in other_arguments for a in common_links]
        # if any(mask):
        #     feature = list(itertools.compress(common_links, mask))

        #2021.02.21 对于单一的feature的处理
        has_single_feature=0
        other_cluster_arg_flat_list = [x for x in data[data[labels] != cluster].arg]
        other_cluster_arg_combine = [item for sublist in other_cluster_arg_flat_list for item in sublist]
        #other_cluster_arg_combine=set(other_cluster_arg_combine)
        for x in common_links:
            if x not in other_cluster_arg_combine:
                feature.append(x)
                has_single_feature=1
        if has_single_feature:
            all_feature.append(str(feature).strip('[]'))






        else:
            other_cluster_arg = [set(x) for x in data[data[labels] != cluster].arg]
            if not any(common_links <= x for x in other_cluster_arg):
                feature.append(common_links)
                for x in range(len(common_links), 1, -1):
                    temp = False
                    combinitions = list(itertools.combinations(common_links, x - 1))  #对于15长度的common_links, 3003 for x =11  太多了
                    for combine in combinitions:
                        combine = set(combine)
                        if not any(combine.issubset(x) for x in other_cluster_arg):
                            temp = True
                            add_to(feature, combine)
                    if not temp:
                        break
            if not feature:
                cluster_with_feature.remove(cluster)
            else:
                all_feature.append(str(feature[0]))

    sum_diff = pd.DataFrame({
        labels: cluster_with_feature,
        "feature_arguments": all_feature,
    })

    return sum_diff
