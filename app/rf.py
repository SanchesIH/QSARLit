import streamlit as st
from st_aggrid import AgGrid

import base64
from io import BytesIO
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
from numpy import sqrt
from numpy import argmax

import pandas as pd

import matplotlib.pyplot as plt

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, roc_curve, roc_auc_score, make_scorer
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
import pickle
from sklearn.calibration import calibration_curve

from imblearn.metrics import geometric_mean_score

from skopt import BayesSearchCV

import plotly.graph_objects as go

def app(df, s_state):
    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
    def getNeighborsDitance(trainingSet, testInstance, k):
        neighbors_k = metrics.pairwise_distances(trainingSet, Y=testInstance, metric='dice', n_jobs=1)
        neighbors_k.sort(0)
        similarity = 1 - neighbors_k
        return similarity[k - 1, :]

    # Cross-validation function
    def cros_val(x, y, classifier, cv):
        probs_classes = []
        y_test_all = []
        AD_fold = []
        distance_train_set = []
        distance_test_set = []
        y_pred_ad = []
        y_exp_ad = []
        for train_index, test_index in cv.split(x, y):
            clf = classifier  # model with best parameters
            X_train_folds = x[train_index]  # descriptors train split
            y_train_folds = np.array(y)[train_index.astype(int)]  # label train split
            X_test_fold = x[test_index]  # descriptors test split
            y_test_fold = np.array(y)[test_index.astype(int)]  # label test split
            clf.fit(X_train_folds, y_train_folds)  # train fold
            y_pred = clf.predict_proba(X_test_fold)  # test fold
            probs_classes.append(y_pred)  # all predictions for test folds
            y_test_all.append(y_test_fold)  # all folds' labels
            k = int(round(pow((len(y)), 1.0 / 3), 0))
            distance_train = getNeighborsDitance(X_train_folds, X_train_folds, k)
            distance_train_set.append(distance_train)
            distance_test = getNeighborsDitance(X_train_folds, X_test_fold, k)
            distance_test_set.append(distance_test)
            Dc = np.average(distance_train) - (0.5 * np.std(distance_train))
            for i in range(len(X_test_fold)):
                ad = 0
                if distance_test_set[0][i] >= Dc:
                    ad = 1
                AD_fold.append(ad)
        probs_classes = np.concatenate(probs_classes)
        y_experimental = np.concatenate(y_test_all)
        # Uncalibrated model predictions
        pred = (probs_classes[:, 1] > 0.5).astype(int)
        for i in range(len(AD_fold)):
            if AD_fold[i] == 1:
                y_pred_ad.append(pred[i])
                y_exp_ad.append(y_experimental[i])

        return pred, y_experimental, probs_classes, AD_fold, y_pred_ad, y_exp_ad

    # STATISTICS
    def calc_statistics(y, pred):
        # save confusion matrix and slice into four pieces
        confusion = confusion_matrix(y, pred)
        # [row, column]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]

        # calc statistics
        accuracy = round(accuracy_score(y, pred), 2)  # accuracy
        mcc = round(matthews_corrcoef(y, pred), 2)  # mcc
        kappa = round(cohen_kappa_score(y, pred), 2)  # kappa
        sensitivity = round(recall_score(y, pred), 2)  # Sensitivity
        specificity = round(TN / (TN + FP), 2)  # Specificity
        positive_pred_value = round(TP / float(TP + FP), 2)  # PPV
        negative_pred_value = round(TN / float(TN + FN), 2)  # NPV
        auc = round(roc_auc_score(y, pred), 2)  # AUC
        bacc = round(balanced_accuracy_score(y, pred), 2)  # balanced accuracy

        # converting calculated metrics into a pandas dataframe to compare all models at the final
        statistics = pd.DataFrame({'Bal-acc': bacc, "Sensitivity": sensitivity, "Specificity": specificity,
                                   "PPV": positive_pred_value,
                                   "NPV": negative_pred_value, 'Kappa': kappa, 'AUC': auc, 'MCC': mcc,
                                   'Accuracy': accuracy, }, index=[0])
        return statistics

    ########################################################################################################################################
    # Seed
    ########################################################################################################################################

    # Choose the general hyperparameters interval to be tested

    with st.sidebar.header('1. Set seed for reproducibility'):
        parameter_random_state = st.sidebar.number_input('Seed number (random_state)', value=42, step=1)

    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################

    # Upload File
    if df is not None:
        # Select activity column
        with st.sidebar.header('2. Select column with activity'):
            name_activity = st.sidebar.selectbox(
                'Select the column with activity (e.g., Active and Inactive that should be 1 and 0, respectively)',
                df.columns)
            if len(name_activity) > 0:
                if name_activity not in df.columns:
                    st.error(f"The column '{name_activity}' is not in the dataframe.")
        st.sidebar.write('---')

        ########################################################################################################################################
        # Data splitting
        ########################################################################################################################################
        with st.sidebar.header('3. Select data splitting'):
            # Select splitting option
            splitting_dict = {'Only k-fold': 'kfold',
                              'k-fold and external set': 'split_original',
                              'Input your own external set': 'input_own', }
            user_splitting = st.sidebar.selectbox('Choose a splitting', list(splitting_dict.keys()))
            selected_splitting = splitting_dict[user_splitting]

        with st.sidebar.subheader('3.1 Number of folds'):
            n_splits = st.sidebar.number_input('Enter the number of folds', min_value=2, max_value=10, value=5)

        # Selecting x and y from input file

        if selected_splitting == 'kfold':
            x = df.drop(columns=[name_activity]).values  # All columns except the activity column
            y = df[name_activity].values  # The activity column

        if selected_splitting == 'split_original':
            x = df.drop(columns=[name_activity]).values  # All columns except the activity column
            y = df[name_activity].values  # The activity column

            with st.sidebar.header('Test size (%)'):
                input_test_size = st.sidebar.number_input('Enter the test size (%)', min_value=1, max_value=99,
                                                          value=20)
                test_size = input_test_size / 100
                x, x_ext, y, y_ext = train_test_split(x, y, test_size=test_size,
                                                      random_state=parameter_random_state, stratify=y)

        if selected_splitting == 'input_own':

            if df is not None:
                # Select activity column
                with st.sidebar.header('2. Select column with activity'):
                    name_activity = st.sidebar.selectbox(
                        'Select the column with activity (e.g., Active and Inactive that should be 1 and 0, respectively)',
                        df.columns)
                    if len(name_activity) > 0:
                        if name_activity not in df.columns:
                            st.error(f"The column '{name_activity}' is not in the dataframe.")
                st.sidebar.write('---')

                x = df.drop(columns=[name_activity]).values  # All columns except the activity column
                y = df[name_activity].values  # The activity column

                x_ext = df_own.drop(columns=[name_activity_ext]).values  # All columns except the activity column
                y_ext = df_own[name_activity_ext].values  # The activity column

    ########################################################################################################################################
    # Sidebar - Specify parameter settings
    ########################################################################################################################################

        st.sidebar.header('5. Set Parameters - Bayesian hyperparameter search')

        # Choose the general hyperparameters
        st.sidebar.subheader('General Parameters')

        parameter_n_iter = st.sidebar.slider('Number of iterations (n_iter)', 1, 1000, 10, 1)
        st.sidebar.write('---')
        parameter_n_jobs = st.sidebar.selectbox('Number of jobs to run in parallel (n_jobs)', options=[-1, 1], index=0)

        # Select the hyperparameters to be optimized
        st.sidebar.subheader('Select the hyperparameters to be optimized')

        container = st.sidebar.container()
        slc_all = st.sidebar.checkbox("Select all")
        rf_hyperparams = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']

        if slc_all:
            selected_options = container.multiselect("Select one or more options:", rf_hyperparams, rf_hyperparams)
        else:
            selected_options = container.multiselect("Select one or more options:", rf_hyperparams)

        # Choose the hyperparameters intervals to be tested
        st.sidebar.subheader('Learning Hyperparameters')

        if not selected_options:
            st.sidebar.write('Please, select the hyperparameters to be optimized!')
        else:
            selected_hyperparameters = {}

            if 'n_estimators' in selected_options:
                st.sidebar.write("Value of estimators (n_estimators)")
                min_n_estimators = st.sidebar.number_input('Min n_estimators', min_value=50, max_value=1000, value=100,
                                                           step=1, key='min_n_estimators')
                max_n_estimators = st.sidebar.number_input('Max n_estimators', min_value=min_n_estimators,
                                                           max_value=1000, value=200, step=1, key='max_n_estimators')
                n_estimators = {'n_estimators': (min_n_estimators, max_n_estimators)}
                selected_hyperparameters.update(n_estimators)
                st.sidebar.write('---')

            if 'max_depth' in selected_options:
                st.sidebar.write("Value of max_depth (max_depth)")
                min_max_depth = st.sidebar.number_input('Min max_depth', min_value=1, max_value=100, value=10, step=1,
                                                        key='min_max_depth')
                max_max_depth = st.sidebar.number_input('Max max_depth', min_value=min_max_depth, max_value=100,
                                                        value=20, step=1, key='max_max_depth')
                max_depth = {'max_depth': (min_max_depth, max_max_depth)}
                selected_hyperparameters.update(max_depth)
                st.sidebar.write('---')

            if 'min_samples_split' in selected_options:
                st.sidebar.write("Value of min_samples_split (min_samples_split)")
                min_samples_split_min = st.sidebar.number_input('Min min_samples_split', min_value=2, max_value=100,
                                                                value=2, step=1, key='min_samples_split_min')
                min_samples_split_max = st.sidebar.number_input('Max min_samples_split',
                                                                min_value=min_samples_split_min, max_value=100,
                                                                value=10, step=1, key='min_samples_split_max')
                min_samples_split = {'min_samples_split': (min_samples_split_min, min_samples_split_max)}
                selected_hyperparameters.update(min_samples_split)
                st.sidebar.write('---')

            if 'min_samples_leaf' in selected_options:
                st.sidebar.write("Value of min_samples_leaf (min_samples_leaf)")
                min_samples_leaf_min = st.sidebar.number_input('Min min_samples_leaf', min_value=1, max_value=100,
                                                               value=1, step=1, key='min_samples_leaf_min')
                min_samples_leaf_max = st.sidebar.number_input('Max min_samples_leaf',
                                                               min_value=min_samples_leaf_min, max_value=100, value=10,
                                                               step=1, key='min_samples_leaf_max')
                min_samples_leaf = {'min_samples_leaf': (min_samples_leaf_min, min_samples_leaf_max)}
                selected_hyperparameters.update(min_samples_leaf)
                st.sidebar.write('---')

    ########################################################################################################################################
    # Modeling
    ########################################################################################################################################

        if st.sidebar.button('Run Modeling'):

            if not selected_hyperparameters:
                st.error("Please select at least one hyperparameter to optimize.")
            else:
                try:
                    # Create folds for cross-validation
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=False, )

                    # Run RF Model building - Bayesian hyperparameter search
                    scorer = make_scorer(geometric_mean_score)

                    opt_rf = BayesSearchCV(
                        RandomForestClassifier(),
                        selected_hyperparameters,
                        n_iter=parameter_n_iter,  # Number of parameter settings that are sampled
                        cv=cv,
                        scoring=scorer,
                        verbose=0,
                        refit=True,  # Refit the best estimator with the entire dataset.
                        random_state=parameter_random_state,
                        n_jobs=parameter_n_jobs
                    )

                    opt_rf.fit(x, y)

                    st.write("Best parameters: %s" % opt_rf.best_params_)

                    # k-fold cross-validation
                    pred_rf, y_experimental, probs_classes, AD_fold, y_pred_ad, y_exp_ad = cros_val(x, y,
                                                                                                   RandomForestClassifier(
                                                                                                       **opt_rf.best_params_),
                                                                                                   cv)
                    # Statistics k-fold cross-validation
                    statistics = calc_statistics(y_experimental, pred_rf)
                    # coverage
                    coverage = round((len(y_exp_ad) / len(y_experimental)), 2)

                    # converting calculated metrics into a pandas dataframe to save a xls
                    model_type = "RF"

                    result_type = "uncalibrated"

                    metrics_rf_uncalibrated = statistics
                    metrics_rf_uncalibrated['model'] = model_type
                    metrics_rf_uncalibrated['result_type'] = result_type
                    metrics_rf_uncalibrated['coverage'] = coverage

                    st.header('**Metrics of uncalibrated model on the K-fold cross-validation**')

                    # Bar chart Statistics k-fold cross-validation

                    metrics_rf_uncalibrated_graph = metrics_rf_uncalibrated.filter(
                        items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC",
                               "coverage"])

                    x_metrics = metrics_rf_uncalibrated_graph.columns
                    y_metrics = metrics_rf_uncalibrated_graph.loc[0].values

                    colors = ["red", "orange", "green", 'yellow', "pink", 'blue', "purple", "cyan", "teal"]

                    fig = go.Figure(data=[go.Bar(
                        x=x_metrics, y=y_metrics,
                        text=y_metrics,
                        textposition='auto',
                        marker_color=colors
                    )])

                    st.plotly_chart(fig)

                    ########################################################################################################################################
                    # External set uncalibrated
                    ########################################################################################################################################
                    if selected_splitting == 'split_original' or selected_splitting == 'input_own':

                        # Predict probabilities for the external set
                        probs_external = opt_rf.predict_proba(x_ext)
                        # Making classes
                        pred_rf_ext = (probs_external[:, 1] > 0.5).astype(int)
                        # Statistics external set uncalibrated
                        statistics_ext = calc_statistics(y_ext, pred_rf_ext)

                        # converting calculated metrics into a pandas dataframe to save a xls
                        model_type = "RF"

                        result_type = "uncalibrated_external_set"

                        metrics_rf_external_set_uncalibrated = statistics_ext
                        metrics_rf_external_set_uncalibrated['model'] = model_type
                        metrics_rf_external_set_uncalibrated['result_type'] = result_type

                        st.header('**Metrics of uncalibrated model on the external set**')
                        # Bar chart Statistics external set

                        metrics_rf_external_set_uncalibrated_graph = metrics_rf_external_set_uncalibrated.filter(
                            items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC",
                                   "coverage"])

                        x_metrics_ext = metrics_rf_external_set_uncalibrated_graph.columns
                        y_metrics_ext = metrics_rf_external_set_uncalibrated_graph.loc[0].values

                        colors = ["red", "orange", "green", 'yellow', "pink", 'blue', "purple", "cyan", "teal"]

                        fig = go.Figure(data=[go.Bar(
                            x=x_metrics_ext, y=y_metrics_ext,
                            text=y_metrics_ext,
                            textposition='auto',
                            marker_color=colors
                        )])

                        st.plotly_chart(fig)

                    ########################################################################################################################################
                    # Model Calibration
                    ########################################################################################################################################
                    # Check model calibration
                    # keep probabilities for the positive outcome only
                    probs = probs_classes[:, 1]
                    # reliability diagram
                    fop, mpv = calibration_curve(y_experimental, probs, n_bins=10)
                    # plot perfectly calibrated
                    fig = plt.figure()
                    plt.plot([0, 1], [0, 1], linestyle='--')
                    # plot model reliability
                    plt.plot(mpv, fop, marker='.')

                    st.header('**Check model calibration**')
                    st.pyplot(fig)

                    # Use ROC-Curve and Gmean to select a threshold for calibration
                    # keep probabilities for the positive outcome only
                    yhat = probs_classes[:, 1]
                    # calculate roc curves
                    fpr, tpr, thresholds = roc_curve(y_experimental, yhat)
                    # calculate the g-mean for each threshold
                    gmeans = sqrt(tpr * (1 - fpr))
                    # locate the index of the largest g-mean
                    ix = argmax(gmeans)
                    # plot the roc curve for the model
                    fig = plt.figure()
                    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
                    plt.plot(fpr, tpr, marker='.', label='RF')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # axis labels
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend()

                    st.header('**Use ROC-Curve and G-Mean to select a threshold for calibration**')

                    st.pyplot(fig)

                    st.write('Best Threshold= %.2f, G-Mean= %.2f' % (round(thresholds[ix], 2), round(gmeans[ix], 2)))

                    # Record the threshold in a variable
                    threshold_roc = round(thresholds[ix], 2)

                    # Select the best threshold to distinguish the classes
                    pred_rf = (probs_classes[:, 1] > threshold_roc).astype(int)

                    # Statistics k-fold cross-validation calibrated
                    statistics = calc_statistics(y_experimental, pred_rf)

                    # Coverage
                    coverage = round((len(y_exp_ad) / len(y_experimental)), 2)

                    # converting calculated metrics into a pandas dataframe to save a xls
                    model_type = "RF"

                    result_type = "calibrated"

                    metrics_rf_calibrated = statistics
                    metrics_rf_calibrated['model'] = model_type
                    metrics_rf_calibrated['result_type'] = result_type
                    metrics_rf_calibrated['calibration_threshold'] = threshold_roc
                    metrics_rf_calibrated['coverage'] = coverage

                    st.header('**Metrics of calibrated model on the K-fold cross-validation**')

                    # Bar chart Statistics k-fold cross-validation calibrated

                    metrics_rf_calibrated_graph = metrics_rf_calibrated.filter(
                        items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC",
                               "coverage"])
                    x_metrics_cal = metrics_rf_calibrated_graph.columns
                    y_metrics_cal = metrics_rf_calibrated_graph.loc[0].values

                    colors = ["red", "orange", "green", 'yellow', "pink", 'blue', "purple", "cyan", "teal"]

                    fig = go.Figure(data=[go.Bar(
                        x=x_metrics_cal, y=y_metrics_cal,
                        text=y_metrics_cal,
                        textposition='auto',
                        marker_color=colors
                    )])

                    st.plotly_chart(fig)

                    ########################################################################################################################################
                    # External set calibrated
                    ########################################################################################################################################
                    if selected_splitting == 'split_original' or selected_splitting == 'input_own':

                        # Predict probabilities for the external set
                        probs_external = opt_rf.predict_proba(x_ext)
                        # Making classes
                        pred_rf_ext = (probs_external[:, 1] > threshold_roc).astype(int)
                        # Statistics external set calibrated
                        statistics_ext = calc_statistics(y_ext, pred_rf_ext)

                        # converting calculated metrics into a pandas dataframe to save a xls
                        model_type = "RF"

                        result_type = "calibrated_external_set"

                        metrics_rf_external_set_calibrated = statistics_ext
                        metrics_rf_external_set_calibrated['model'] = model_type
                        metrics_rf_external_set_calibrated['result_type'] = result_type

                        st.header('**Metrics of calibrated model on the external set**')
                        # Bar chart Statistics external set

                        metrics_rf_external_set_calibrated_graph = metrics_rf_external_set_calibrated.filter(
                            items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC",
                                   "coverage"])

                        x_metrics_ext_cal = metrics_rf_external_set_calibrated_graph.columns
                        y_metrics_ext_cal = metrics_rf_external_set_calibrated_graph.loc[0].values

                        colors = ["red", "orange", "green", 'yellow', "pink", 'blue', "purple", "cyan", "teal"]

                        fig = go.Figure(data=[go.Bar(
                            x=x_metrics_ext_cal, y=y_metrics_ext_cal,
                            text=y_metrics_ext_cal,
                            textposition='auto',
                            marker_color=colors
                        )])

                        st.plotly_chart(fig)

                    ########################################################################################################################################
                    # Compare models
                    ########################################################################################################################################

                    # Only K-fold
                    st.header('**Compare metrics of calibrated and uncalibrated models on the K-fold cross-validation**')

                    metrics_rf_uncalibrated_graph = metrics_rf_uncalibrated.filter(
                        items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC"])
                    metrics_rf_calibrated_graph = metrics_rf_calibrated.filter(
                        items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC"])

                    fig = go.Figure()

                    fig.add_trace(go.Scatterpolar(
                        r=metrics_rf_uncalibrated_graph.loc[0].values,
                        theta=metrics_rf_uncalibrated_graph.columns,
                        fill='toself',
                        name='Uncalibrated'
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=metrics_rf_calibrated_graph.loc[0].values,
                        theta=metrics_rf_uncalibrated_graph.columns,
                        fill='toself',
                        name='Calibrated'
                    ))

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True
                    )

                    st.plotly_chart(fig)

                    # External set

                    if selected_splitting == 'split_original' or selected_splitting == 'input_own':

                        st.header(
                            '**Compare metrics of calibrated and uncalibrated models on the external set**')

                        metrics_rf_external_set_uncalibrated_graph = metrics_rf_external_set_uncalibrated.filter(
                            items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC"])
                        metrics_rf_external_set_calibrated_graph = metrics_rf_external_set_calibrated.filter(
                            items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC"])

                        fig = go.Figure()

                        fig.add_trace(go.Scatterpolar(
                            r=metrics_rf_external_set_uncalibrated_graph.loc[0].values,
                            theta=metrics_rf_external_set_uncalibrated_graph.columns,
                            fill='toself',
                            name='Uncalibrated'
                        ))
                        fig.add_trace(go.Scatterpolar(
                            r=metrics_rf_external_set_calibrated_graph.loc[0].values,
                            theta=metrics_rf_external_set_uncalibrated_graph.columns,
                            fill='toself',
                            name='Calibrated'
                        ))

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=True
                        )

                        st.plotly_chart(fig)

                    ########################################################################################################################################
                    # Download files
                    ########################################################################################################################################

                    st.header('**Download files**')

                    if selected_splitting == 'split_original' or selected_splitting == 'input_own':
                        frames = [metrics_rf_uncalibrated, metrics_rf_calibrated,
                                  metrics_rf_external_set_uncalibrated, metrics_rf_external_set_calibrated]

                    else:
                        frames = [metrics_rf_uncalibrated, metrics_rf_calibrated, ]

                    result = pd.concat(frames)

                    result = result.round(2)

                    # File download
                    def filedownload(df):
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
                        href = f'<a href="data:file/csv;base64,{b64}" download="metrics_rf.csv">Download CSV File - metrics</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    filedownload(result)

                    def download_model(model):
                        output_model = pickle.dumps(model)
                        b64 = base64.b64encode(output_model).decode()
                        href = f'<a href="data:file/output_model;base64,{b64}" download="model_rf.pkl">Download generated model (PKL File)</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    download_model(opt_rf)

                except Exception as e:
                    st.error(f"An error occurred during modeling: {e}")
