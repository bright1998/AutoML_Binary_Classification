import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

def AML_BinaryClassification(mfile, sfile, header_ID, header_obj, ohe_list, _PCA, scoring_method, n_GS=3, cls=1):
# ファイル読み込み時のデータ型指定用辞書作成
    dtype_dict = {}
    for item in ohe_list:
        dtype_dict[item] = 'object'

# データファイルの読み込み
# データフレーム
    df_m = pd.read_csv(mfile, dtype=dtype_dict)
    df_s = pd.read_csv(sfile, dtype=dtype_dict)

# IDの抽出
    ID_m = df_m[[header_ID]]
    ID_s = df_s[[header_ID]]

# 説明変数と目的変数の抽出
    y_m = df_m[header_obj]
    X_m = df_m.drop([header_ID, header_obj], axis=1)

    try:
        y_s = df_s[header_obj]
        X_s = df_s.drop([header_ID, header_obj], axis=1)
    except:
        X_s = df_s.drop([header_ID], axis=1)

##### 本番は不要 #####
# 目的変数の数値値
#    class_mapping = {'N':1, 'Y':0}
#    y_m = y_m.map(class_mapping)
#####################

# one-hotエンコーディング & 欠損値処理
# モデル用
    X_ohe_m = pd.get_dummies(X_m, dummy_na=True, columns=ohe_list)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X_ohe_m)
    X_ohe2_m = pd.DataFrame(data=imp.transform(X_ohe_m), columns=X_ohe_m.columns.values)

# スコア用
    X_ohe_s = pd.get_dummies(X_s, dummy_na=True, columns=ohe_list)
    X_ohe2_s = pd.DataFrame(data=None, columns=X_ohe2_m.columns.values)
    X_ohe2_s = pd.concat([X_ohe2_s, X_ohe_s])
    X_ohe2_s.loc[:, list(set(X_ohe_m.columns) - set(X_ohe_s.columns))] = X_ohe2_s.loc[:, list(set(X_ohe_m.columns) - set(X_ohe_s.columns))].fillna(value=0, axis=1)
    X_ohe3_s = X_ohe2_s.drop(list(set(X_ohe_s.columns) - set(X_ohe_m.columns)), axis=1)
    X_ohe4_s = X_ohe3_s.reindex(columns=X_ohe2_m.columns.values)
    X_ohe5_s = pd.DataFrame(data=imp.transform(X_ohe4_s), columns=X_ohe4_s.columns.values)

# 特徴量選択（RFECV）
    selector = RFECV(RandomForestClassifier(random_state=1), step=0.05, scoring=scoring_method)
    selector.fit(X_ohe2_m.values, y_m.values)
    X_fin_m = X_ohe2_m.loc[:, X_ohe2_m.columns.values[selector.support_]]
    X_fin_s = X_ohe5_s.loc[:, X_ohe5_s.columns.values[selector.support_]]
    display(X_fin_m.shape)

# モデル用データの分割
    X_train_eval, X_test, y_train_eval, y_test = train_test_split(X_fin_m, y_m, train_size=0.8, test_size=0.2, random_state=1)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train_eval, y_train_eval, train_size=0.8, test_size=0.2, random_state=1)

#　モデル選択（K-hold法)
    if use_PCA == True:
        pipelines = {
# ロジスティック回帰
            'logistic' : Pipeline([('scl', StandardScaler()), ('PCA', PCA(random_state=1)), ('est', LogisticRegression(random_state=1))]),
## サポートベクターマシン
#            'svm' : Pipeline([('scl', StandardScaler()), ('PCA', PCA(random_state=1)), ('est', LinearSVC(random_state=1))]),
# k近傍法（n_neighbors=5）
            'knn' : Pipeline([('scl', StandardScaler()), ('PCA', PCA(random_state=1)), ('est', KNeighborsClassifier())]),
# ランダムフォレスト（n_estimators=10）
            'rfc' : Pipeline([('scl', StandardScaler()), ('PCA', PCA(random_state=1)), ('est', RandomForestClassifier(random_state=1))]),
# 勾配ブースティング（n_estimators=100）
            'gbc' : Pipeline([('scl', StandardScaler()), ('PCA', PCA(random_state=1)), ('est', GradientBoostingClassifier(random_state=1))]),
#            'gbc' : Pipeline([('scl', StandardScaler()), ('PCA', PCA(random_state=1)), ('est', GradientBoostingClassifier(random_state=1))])
# 多層パーセプトロン（hidden_layer_sizes=(100,), activation='relu', solver='adam'）
            'mlp' : Pipeline([('scl', StandardScaler()), ('PCA', PCA(random_state=1)), ('est', MLPClassifier(max_iter=10000, random_state=1))])
#            '' : Pipeline([('scl', StandardScaler()), ('PCA', PCA(random_state=1)), ('est', )])
        }

    else:
        pipelines = {
# ロジスティック回帰
            'logistic' : Pipeline([('scl', StandardScaler()), ('est', LogisticRegression(random_state=1))]),
## サポートベクターマシン
#            'svm' : Pipeline([('scl', StandardScaler()), ('est', LinearSVC(random_state=1))]),
# k近傍法（n_neighbors=5）
            'knn' : Pipeline([('scl', StandardScaler()), ('est', KNeighborsClassifier())]),
# ランダムフォレスト（n_estimators=10）
            'rfc' : Pipeline([('scl', StandardScaler()), ('est', RandomForestClassifier(random_state=1))]),
# 勾配ブースティング（n_estimators=100）
            'gbc' : Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=1))]),
#            'gbc' : Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=1))])
# 多層パーセプトロン（hidden_layer_sizes=(100,), activation='relu', solver='adam'）
            'mlp' : Pipeline([('scl', StandardScaler()), ('est', MLPClassifier(max_iter=10000, random_state=1))])
#            '' : Pipeline([('scl', StandardScaler()), ('est', )])
        }

# 交差検定
    results = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for pipe_name, pipeline in pipelines.items():
        cv_results = cross_val_score(estimator=pipeline, X=X_train.values, y=y_train.values, scoring=scoring_method, cv=kf)
        print(pipe_name, ':', scoring_method, 'score= ', cv_results.mean(), '+-', cv_results.std())
#        print(pipe_name, ':', scoring_method, 'score= ', cv_results.mean() - cv_results.std())
        results[pipe_name] = cv_results.mean() - cv_results.std()

# ランキング（評価値の高い順にソート）
    sorted_results = {}
    for k,v in sorted(results.items(), key=lambda x:-x[1]):
        sorted_results[k] = v

    df_ranking = pd.DataFrame(data=list(sorted_results.values()), index=list(sorted_results.keys()), columns=['score'])
    df_ranking.to_csv('model_ranking.csv')
    print('')
    print('Result of K-Fold (Cross Validation)')
    display(df_ranking)

# トップからn_GS個の手法をグリッドサーチでパラメータチューニング
    best_scores = {}
    best_models = {}
    for i in range(n_GS):
        method_name = list(sorted_results.keys())[i]
        if method_name == 'logistic':
            param_grid = {'est__penalty':['l1', 'l2'], 'est__C':[0.1, 0.5, 1, 2, 5, 10]}
        elif method_name == 'svm':
            param_grid = {'est__loss':['hinge', 'squared_hinge'], 'est__C':[0.1, 1, 10]}
        elif method_name == 'knn':
            param_grid = {'est__n_neighbors':[2, 5, 10, 15], 'est__weights':['uniform', 'distance']}
#            param_grid = {'est__n_neighbors':np.random.randint(low=2, high=16, size=4), 'est__weights':['uniform', 'distance']}
        elif method_name == 'rfc':
            param_grid = {'est__n_estimators':[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'est__max_features':['auto', 'log2']}
        elif method_name == 'gbc':
            param_grid = {'est__n_estimators':[30, 50, 70, 100, 150, 200], 'est__max_depth':[3, 4, 5, 6, 7, 8, 9, 10]}
        elif method_name == 'mlp':
            param_grid = {'est__hidden_layer_sizes':[(100,), (100, 100,), (100, 100, 100,)], 'est__activation':['relu', 'logistic'], 'est__solver':['adam', 'lbfgs', 'sgd']}
        gs = GridSearchCV(pipelines[method_name], param_grid, scoring=scoring_method, cv=kf)
        gs.fit(X_eval.values, y_eval.values)
        best_scores[method_name] = gs.best_score_
        best_models[method_name] = gs.best_estimator_
#        display('gs.best_estimator_= ', gs.best_estimator_)
#        display('gs.best_params_= ', gs.best_params_)
    print('===Result of Grid Search===')
    print(best_scores)
#    print(best_models)
    print('')

# ランキング（評価値の高い順にソート）
    sorted_best_scores = {}
    for k,v in sorted(best_scores.items(), key=lambda x:-x[1]):
        sorted_best_scores[k] = v
    print('Best Model:', list(sorted_best_scores.keys())[0])
    print('')

# ベストモデルの抽出
    best_model = best_models[list(sorted_best_scores.keys())[0]]
    display(best_model)

# ベストモデルの学習
    best_model.fit(X_train_eval.values, y_train_eval.values)
    print('=== Fitting Done ===')
    print('')

# 学習済みモデルの評価（汎化性能の確認）
    print('=== Checking Generalization Ability ===')
    if scoring_method == 'accuracy':
        print('Done:', scoring_method, 'score=', accuracy_score(y_test.values, best_model.predict(X_test)))
    elif scoring_method == 'precision':
        print('Done:', scoring_method, 'score=', precision_score(y_test.values, best_model.predict(X_test)))
    elif scoring_method == 'recall':
        print('Done:', scoring_method, 'score=', recall_score(y_test.values, best_model.predict(X_test)))
    elif scoring_method == 'f1':
        print('Done:', scoring_method, 'score=', f1_score(y_test.values, best_model.predict(X_test)))
    elif scoring_method == 'roc_auc':
        df_prediction_class = pd.DataFrame(data=best_model.predict(X_test), columns=['Y_pred'])
        prediction_proba = best_model.predict_proba(X_test)
        for i in range(len(df_prediction_class)):
            if df_prediction_class.loc[i, 'Y_pred'] == cls:
                icheck = i
                break
        check_list = list(prediction_proba[icheck, :])
        max_proba = 0
        for i in range(len(check_list)):
            if check_list[i] > max_proba:
                max_proba = check_list[i]
                imax = i
        y_score = best_model.predict_proba(X_test)[:, i]
        print('Done:', scoring_method, 'score=', roc_auc_score(y_test.values, y_score))
#    elif scoring_method == '':

# モデルの保存
    with open('best_model.pickle', mode='wb') as fp:
        pickle.dump(best_model, fp, protocol=2)

# モデルの読み込み
#    with open('best_model.pickle', mode='rb') as fp:
#        best_model = pickle.load(fp)

# スコア用データでの予測
    df_prediction_class = pd.DataFrame(data=best_model.predict(X_fin_s), columns=['Y_pred'])

    prediction_proba = best_model.predict_proba(X_fin_s)
    for i in range(len(df_prediction_class)):
        if df_prediction_class.loc[i, 'Y_pred'] == cls:
            icheck = i
            break
    check_list = list(prediction_proba[icheck, :])
    max_proba = 0
    for i in range(len(check_list)):
        if check_list[i] > max_proba:
            max_proba = check_list[i]
            imax = i

    df_prediction_proba = pd.DataFrame(data=prediction_proba[:,imax], columns=['Y_proba'])

#    result_fin = pd.concat([ID_s, df_prediction_class, df_prediction_proba], axis=1)
    result_fin = pd.concat([ID_s, df_prediction_proba], axis=1)
    result_fin = result_fin.set_index(header_ID)
    display(result_fin.head())
    display(result_fin.shape)
    result_fin.to_csv('predict_probability.csv')

model_file_name = 'data/train.csv'
score_file_name = 'data/test.csv'
header_ID = 'ID'
header_obj = 'Objective'
ohe_list = ['x1', 'x2']
use_PCA = True
#scoring_method = 'f1'
#scoring_method = 'accuracy'
scoring_method = 'roc_auc'
AML_BinaryClassification(model_file_name, score_file_name, header_ID, header_obj, ohe_list, use_PCA, scoring_method, n_GS=3, cls=1)
