import pandas as pd
import plotly.express as px
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from plotly.offline import init_notebook_mode


def read_data(file_path):
    return pd.read_csv(file_path)


def print_target_distribution(data, target_name, notebook_mode=False):
    if notebook_mode:
        init_notebook_mode(connected=True)

    mutation_counts = data[target_name].value_counts()
    fig = px.bar(mutation_counts, x=mutation_counts.index, y=mutation_counts.values, title='Class distribution')
    fig.show()


def print_basic_statistics(data, target_name):
    pd.set_option('display.width', 1000)
    print(f'Shape = {data.shape}\n')
    print(f'Columns = {data.columns.values}\n')
    print(f'First 15 samples:\n{data.head(15)}\n')
    print(f'Basic statistics:\n{data.describe().T}\n')
    print_target_distribution(data, target_name)


def show_data_missings(data):
    msno.matrix(data)
    plt.figure(figsize = (15,9))
    plt.show()


def label_encode_string_values(data):
    data_encoded = data.copy()
    encoder = LabelEncoder()

    for column in data.columns:
        if data_encoded[column].dtype == 'object':
            data_encoded[column] = encoder.fit_transform(data[column])

    return data_encoded


def get_string_columns(data):
    string_columns = []

    for column in data.columns:
        if data[column].dtype == 'object':
            string_columns.append(column)

    return string_columns


def get_numerical_columns(data):
    numerical_columns = []

    for column in data.columns:
        if data[column].dtype != 'object':
            numerical_columns.append(column)

    return numerical_columns


def onehot_encode_string_values(data):
    data_encoded = data.copy()
    string_columns = get_string_columns(data)

    for column in string_columns:
        df_encoded_column = pd.get_dummies(data[column], drop_first=True, prefix=column).astype(int)

        data_encoded = pd.concat([data_encoded.drop(column, axis=1), df_encoded_column], axis=1)            

    return data_encoded


def std_scale_numerical_values(data):
    data_scaled = data.copy()
    scaler_class = StandardScaler
    numerical_columns = get_numerical_columns(data)

    for column in numerical_columns:
        data_scaled[column] = scaler_class().fit_transform(data_scaled[column])
    
    return data_scaled


def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = build_confusion_matrix(y_true, y_pred)

    plt.figure(figsize = (5,5))
    sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu", cbar=False);
    return plt


def build_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    return pd.DataFrame(data = cm, columns = ['Predicted:0','Predicted:1'], index=['Actual:0','Actual:1'])


def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    plt.plot(fpr, tpr, label='ROC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC: {:.3f}'.format(auc_score))
    plt.show()


def plot_roc_curves(models, x_test, y_test, figsize):
    plt.figure(figsize=figsize)

    for model in models:
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Different Models')
    plt.legend(loc='lower right')

    plt.show()


def plot_corr_matrix(data, figsize):
    plt.figure(figsize=figsize)
    sns.heatmap(data.corr(), annot=True);


def pair_plot(data, target):
    sns.pairplot(data, hue=target);


def get_basic_quality_metrics_(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return accuracy, precision, recall, f1


def get_basic_quality_metrics(y_true, y_pred, y_pred_proba):
    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return roc_auc, accuracy, precision, recall, f1


def print_basic_quality_metrics_(y_true, y_pred):
    accuracy, precision, recall, f1 = get_basic_quality_metrics_(y_true, y_pred)
    
    print("Accuracy:  {:.3f}\nPrecision: {:.3f}\nRecall:    {:.3f}\nF1-score:  {:.3f}".format(
        accuracy, precision, recall, f1
    ))


def print_basic_quality_metrics(y_true, y_pred, y_pred_proba):
    roc_auc, accuracy, precision, recall, f1 = get_basic_quality_metrics(y_true, y_pred, y_pred_proba)
    
    print("ROC AUC:   {:.3f}\nAccuracy:  {:.3f}\nPrecision: {:.3f}\nRecall:    {:.3f}\nF1-score:  {:.3f}".format(
        roc_auc, accuracy, precision, recall, f1
    ))


def print_metrics_table(metrics, rows_names=None):
    df = pd.DataFrame(data=metrics, columns=['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1-score'], index=rows_names)
    print(df)


def render_plots_table(plots, num_cols_to_display, figsize=None, plots_names=None):
    num_rows = (len(plots) - 1) // num_cols_to_display + 1

    fig, axes = plt.subplots(num_rows, num_cols_to_display, figsize=figsize)

    axes = axes.flatten()

    for i, plot in enumerate(plots):
        row = i % num_rows
        col = i % num_cols_to_display
        axes[i].imshow(plot)
        axes[i].set_title(plots_names[i])

    plt.tight_layout()
    plt.show()


def basic_model_test(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print_basic_quality_metrics(y_test, y_pred, y_pred_proba)


def ndarray_to_dataframe(arr, columns=None):
    arr_index = range(0, len(arr))

    df = pd.DataFrame(data=arr, index=arr_index, columns=columns)
    df = df.rename_axis('ID')

    return df


def ndarray_to_csv(arr, path):
    dataframe = ndarray_to_dataframe(arr)

    dataframe.to_csv(path)


def plot_explained_variance_ratio(explained_variance_ratio, algo_name):
    plt.bar(range(0, len(explained_variance_ratio)), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Explained Variance Ratio of {algo_name}')
    plt.show()


def plot_cumulative_explained_variance_ratio(explained_variance_ratio, algo_name, threshold):
    explained_variance_ratio = explained_variance_ratio.copy()
    enough_features = -1

    for i in range(1, len(explained_variance_ratio)):
        explained_variance_ratio[i] += explained_variance_ratio[i-1]
        if (enough_features == -1):
            if (explained_variance_ratio[i] >= threshold):
                enough_features = i + 1
                print(f'First {enough_features} features explain enough variance for threshold {threshold}: {explained_variance_ratio[i]}')
     
    plt.bar(range(0, len(explained_variance_ratio)), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'Cumulative Explained Variance Ratio of {algo_name}')
    plt.axhline(y=threshold, color='red', linestyle='--')

    plt.show()

    return enough_features


def corr_feature_detect(data, threshold=0.8):
    corrmat = data.corr()
    corrmat = corrmat.unstack()
    corrmat = corrmat.sort_values(ascending=False)
    corrmat = corrmat[(corrmat >= threshold) | (corrmat <=-threshold)]
    corrmat = corrmat[corrmat < 1]
    corrmat = pd.DataFrame(corrmat).reset_index()
    corrmat.columns = ['feature1', 'feature2', 'corr']

    grouped_features_list = []
    correlated_groups = []

    for feature in corrmat.feature1.unique():
        if feature is not grouped_features_list:
            correlated_block = corrmat[corrmat.feature1 == feature]
            grouped_features_list = grouped_features_list + list(correlated_block.feature2.unique()) + [feature]
            correlated_groups.append(correlated_block)
    return correlated_groups


def outlier_detect_IQR(dataset, col, threshold, print_info=False):
    IQR = dataset[col].quantile(0.75) - dataset[col].quantile(0.25)
    lower_fence = dataset[col].quantile(0.25) - (IQR * threshold)
    upper_fence = dataset[col].quantile(0.75) + (IQR * threshold)
    param = (upper_fence, lower_fence)
    tmp = pd.concat([dataset[col]>upper_fence, dataset[col]<lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)

    return outlier_index, param


def outlier_detect_mean_std(dataset, col, threshold, print_info=False):
    lower_fence = dataset[col].mean() - threshold * dataset[col].std()
    upper_fence = dataset[col].mean() + threshold * dataset[col].std()
    param = (upper_fence, lower_fence)
    tmp = pd.concat([dataset[col]>upper_fence, dataset[col]<lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    
    return outlier_index, param


def print_outlier_detect_summary(outlier_index):
    if True in outlier_index.value_counts().index:
        print("Number of outliers: ", outlier_index.value_counts()[1])
        print("Outlier share: ", outlier_index.value_counts()[1]/len(outlier_index))
    else:
        print("No outliers!")


def outlier_detect(dataset, col, threshold, method='std', print_summary=False):
    if method == 'IQR':
        outlier_index, param = outlier_detect_IQR(dataset, col, threshold)
    else:
        outlier_index, param = outlier_detect_mean_std(dataset, col, threshold)

    if print_summary:
        print_outlier_detect_summary(outlier_index)

    return outlier_index, param


def concat_outliers(outliers):
    df = pd.DataFrame()

    for _, outliers in outliers.items():
        outlier_index, param = outliers
        df = pd.concat([df, outlier_index], axis=1)

    return df.any(axis=1)


def save_pyobj_to_file(pyobj, file_path):
    joblib.dump(pyobj, file_path)


def load_pyobj_from_file(file_path):
    return joblib.load(file_path)


def plot_features_distributions(data, columns_to_plot, num_cols_to_display):
    num_rows = (len(columns_to_plot) - 1) // num_cols_to_display + 1

    fig, axes = plt.subplots(num_rows, num_cols_to_display, figsize=(25, 20))

    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        sns.histplot(data=data, x=col, kde=True, ax=axes[i])
        axes[i].set_title(f"Histogram for {col}")
        axes[i].set_xlabel(col)

    plt.tight_layout()
    plt.show()


def plot_conf_matrix_table(conf_matrices, num_cols, figsize, titles):
    num_rows = (len(conf_matrices) - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, conf_matrix in enumerate(conf_matrices):
        ax = axes[i] if num_cols > 1 else axes
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", cbar=False, ax=ax)
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()

