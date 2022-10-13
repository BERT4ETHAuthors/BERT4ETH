import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, roc_curve, auc, precision_recall_curve
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("algo", "bert4eth", "algorithm for embedding generation" )

def load_embedding():

    exp_addr_set = set(np.load("bert4eth/data/exp_addr.npy"))

    if FLAGS.algo == "bert4eth":
        embeddings = np.load("bert4eth/data/xx.npy")
        address_for_embedding = np.load("bert4eth/data/xx.npy")

    else:
        raise ValueError("should choose right algo..")

    # group by embedding according to address
    address_to_embedding = {}
    for i in range(len(address_for_embedding)):
        address = address_for_embedding[i]
        embedding = embeddings[i]
        try:
            address_to_embedding[address].append(embedding)
        except:
            address_to_embedding[address] = [embedding]

    # group to one
    address_list = []
    embedding_list = []

    for addr, embeds in address_to_embedding.items():
        if addr not in exp_addr_set:
            continue
        address_list.append(addr)
        if len(embeds) > 1:
            embedding_list.append(np.mean(embeds, axis=0))
        else:
            embedding_list.append(embeds[0])

    # final embedding table
    X = np.array(np.squeeze(embedding_list))

    return X, address_list

def main():

    phisher_account = pd.read_csv("../Data/phisher_account.txt", names=["account"])
    phisher_account_set = set(phisher_account.account.values)

    X, address_list = load_embedding()

    y = []
    for addr in address_list:
        if addr in phisher_account_set:
            y.append(1)
        else:
            y.append(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # model = LogisticRegression(random_state=345, C=0.08, max_iter=500)
    # model.fit(X_train, y_train)

    model = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
    model.fit(X_train, y_train)

    print("=============Precision-Recall Curve=============")
    y_test_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_proba)
    plt.figure("P-R Curve")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall, precision)
    plt.show()

    print("================ROC Curve====================")
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba, pos_label=1)
    print("AUC=", auc(fpr, tpr))
    plt.figure("ROC Curve")
    plt.title("ROC Curve")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.plot(fpr, tpr)
    plt.show()

    print("pause")

    # # visualization
    # y = np.array(y)
    # p_idx = np.where(y == 1)[0]
    # n_idx = np.where(y == 0)[0]
    # X_phisher = X[p_idx]
    # X_normal = X[n_idx]
    #
    # permutation = np.random.permutation(len(X_normal))
    # X_normal_sample = X_normal[permutation[:10000]]
    # X4tsne = np.concatenate([X_normal_sample, X_phisher], axis=0)
    # tsne = TSNE(n_components=2, init="random")
    # X_tsne = tsne.fit_transform(X4tsne)
    #
    # # plot
    # plt.figure(figsize=(8, 6), dpi=80)
    # plt.scatter(x=X_tsne[:10000, 0], y=X_tsne[:10000, 1], marker=".")
    # plt.scatter(x=X_tsne[10000:, 0], y=X_tsne[10000:, 1], marker=".", color="orange")
    # plt.show()
    #
    # plt.figure(figsize=(8, 6), dpi=80)
    # plt.scatter(x=X_tsne[:10000, 0], y=X_tsne[:10000, 1], marker=".")
    # plt.show()


if __name__ == '__main__':
    main()
