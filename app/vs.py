
########################################################################################################################################
# Credits
########################################################################################################################################

# Developed by José Teófilo Moreira Filho, Ph.D.
# teofarma1@gmail.com
# http://lattes.cnpq.br/3464351249761623
# https://www.researchgate.net/profile/Jose-Teofilo-Filho
# https://scholar.google.com/citations?user=0I1GiOsAAAAJ&hl=pt-BR
# https://orcid.org/0000-0002-0777-280X

########################################################################################################################################
# Importing packages
########################################################################################################################################

#from st_aggrid import AgGrid
import streamlit as st

import base64
import warnings
warnings.filterwarnings(action='ignore')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

#from rdkit.Chem import PandasTools

import numpy as np

import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import pickle

#from rdkit.Chem.Draw import DrawingOptions

import utils
########################################################################################################################################
# Page Title
########################################################################################################################################

def app(df,s_state):

    info={
        "PKL":"A PKL file is a file generated by pickle method, you can generate it in the machine learning methods in the app, or upload one of your own", 
        "CSV":"Awaiting for CSV file to be uploaded.",
        "Modeling":["Awaiting for Modeling Data file to be uploaded.","If you have SMILES ***AND*** want to see the molecules: name the column with the SMILES and save the SMILES column"],
        "Links":["https://github.com/joseteofilo/data_qsarlit/blob/master/descriptor_morgan_r2_2048bits_for_modeling.csv", "https://github.com/joseteofilo/data_qsarlit/blob/master/descriptor_morgan_r2_2048bits_for_vs.csv", "https://github.com/joseteofilo/data_qsarlit/blob/master/model_rf_morgan_r2_2048bits.pkl"]
        }
    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
    #repeated code
    cc = utils.Custom_Components()
    def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="virtual_sreening_results_rf.csv">Download CSV File - Predictions</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def persist_dataframe(df,updated_df,col_to_delete):
            """Pass 2 keys for both aggrid objects"""
            """df is the original dataframe"""
            delete_col = st.session_state[col_to_delete]
            #deleted_columns=[delete_col for delete_col in st.session_state.col_to_delete]
            
            if delete_col in st.session_state[updated_df]:
                #deleted_col=pd.DataFrame(st.session_state.updated_df[delete_col])
                st.session_state[updated_df] = st.session_state[updated_df].drop(columns=[delete_col])
            
            else:
                st.sidebar.warning("Column previously deleted. Select another column.")
            
            with st.container():
                st.header("**Updated input data**") 
                st.write(st.session_state[updated_df])
                st.header('**Original input data**')
                st.write(df)
            

    def getNeighborsDitance(trainingSet, testInstance, k):
        neighbors_k=metrics.pairwise.pairwise_distances(trainingSet, Y=testInstance, metric='dice', n_jobs=1)
        neighbors_k.sort(0)
        similarity= 1-neighbors_k
        return similarity[k-1,:]

    def saveDFs(df,expander_name,updated_df,column_to_delete):
        """Pass keys for st.write unique widget"""
        with st.expander(expander_name):
                persist_dataframe(df,updated_df,column_to_delete)
        
    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################

    # Upload File for predictions
    #repeated code
    
    # with st.sidebar.header('1. Upload your CSV of modeling data (AD calculation)'):
    #     uploaded_file = st.sidebar.file_uploader("Upload your CSV data for predictions", type=["csv"], key="up_CSV_VS")
    # st.sidebar.markdown("[Example CSV input file](%s)"% info["Links"][0])
    #repeated code
    if df is not None:
        # Read CSV data
        #f = pd.read_csv(uploaded_file, sep=',')
        droped = cc.delete_column(df)
    #     if "updated_df" not in st.session_state:
    #         st.session_state.updated_df = df
    #         index_csv=0
    #         st.header('**Original input data**')
    #         st.write(df)
        
    #     st.sidebar.header("Please delete columns not related to descriptor!")
    #     if "updated_df" in st.session_state:
    #         with st.sidebar.form("descriptor_form"):
    #             if len(st.session_state["updated_df"].columns.tolist())>0:
    #                 for i in range(len(st.session_state["updated_df"].columns.tolist())):
    #                     if i is not None:
    #                         index_csv=i
    #                         break
    #                 index = df.columns.tolist().index(
    #                 st.session_state["updated_df"].columns.tolist()[index_csv]
    #             )
    #             st.selectbox(
    #                 "Select column to delete", options=df.columns, index=index, key="delete_col"
    #             )
    #             delete = st.form_submit_button(label="Delete")
    #         if delete:
    #             # Persist dataframe and show it inside an expander
    #             if "updated_modeling_df" in st.session_state:
    #                 # show Both if user inputs first df
    #                 with st.expander("Modeling"):
    #                     st.write(st.session_state.updated_modeling_df,key="inside_org_csv_expander")
    #                 saveDFs(df,"Predictions","updated_df","Sdelete_col")
    #             else:
    #                 saveDFs(df,"Predictions","updated_df","delete_col")
    # else:
    #     st.info("Awaiting for CSV file to be uploaded.")
  
    #-----------------------------------------------------------------------------------------------------#

    # Select columns
    
    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################

    # Upload File for predictions
    # repeated code
    #with st.sidebar.header('2. Upload your CSV for predictions '):
    uploaded_file_modeling = cc.upload_file(custom_title="Upload your input CSV of modeling data (AD calculation)", context=st.sidebar, key="up_AD_VS")
    cc.delete_column(df,key="upload_file_modeling")
    #st.sidebar.markdown("[Example CSV input file](%s)"% info["Links"][1])
    
    # with st.sidebar.container():
    #     st.sidebar.header("Please delete columns not related to descriptor!")
    """with st.sidebar.expander("If you want to see your MOLECULES"):
        st.write(info["Modeling"][1])
    name_smiles_vs = st.sidebar.text_input('Enter column name with SMILES', 'SMILES')
    save_smiles=st.sidebar.button("Save Smiles")"""
    #repeated code
    if  uploaded_file_modeling is not None:
        # Read CSV data
        df_modeling = pd.read_csv(uploaded_file_modeling, sep=',')
        original_model = df_modeling.copy()
        
        """if save_smiles:
            
            df_modeling=pd.DataFrame(df_modeling)
            
            smiles_column=pd.Series(df_modeling[name_smiles_vs]) 
            st.write(smiles_column)
        """
        # if "updated_modeling_df" not in st.session_state:
        #     st.session_state.updated_modeling_df = df_modeling
        #     index_modeling=0   
        #     if df is not None and "updated_df" not in st.session_state:
        #         st.session_state.updated_df = df
        #     if "updated_df" in st.session_state:
        #         st.session_state.updated_df = st.session_state.updated_df
        #         with st.expander("Prediction"):
        #             st.write(st.session_state.updated_df)
            
        #     st.header('**Original input data**')
        #     st.write(df_modeling)
            
        # If the modeling df is in session state compute rest
        
    #     if "updated_modeling_df" in st.session_state:
    #         if st.session_state.updated_modeling_df.columns.tolist() not in df_modeling.columns.tolist():
    #             st.write(df_modeling,key="test")
    #         with st.sidebar.form("modeling_form"):
    #                 if len(st.session_state.updated_modeling_df.columns.tolist())>0:
    #                     for i in range(len(st.session_state.updated_modeling_df.columns.tolist())):
    #                         if i is not None:
    #                             index_modeling=i
    #                             break
    #                     try:
    #                         modeling_index = df_modeling.columns.tolist().index(
    #                         st.session_state.updated_modeling_df.columns.tolist()[index_modeling]
    #                         )
    #                     except ValueError:
    #                         modeling_index = df_modeling.columns.tolist().index(
    #                         df_modeling.columns.tolist()[0]
    #                         )
    #                 st.selectbox("Select column to delete", options=st.session_state.updated_modeling_df.columns, index=modeling_index, key="delete_modeling_col")
    #                 delete_modeling = st.form_submit_button(label="Delete")
    #         if delete_modeling:
    #             if "updated_df" in st.session_state:
    #                 # show Both if user inputs first df
    #                 with st.expander("Predictions"):
    #                     st.write(st.session_state.updated_df,key="inside_modeling_expander")
    #                 saveDFs(df_modeling,"Modeling","updated_modeling_df","delete_modeling_col")
    #             else:
    #                 saveDFs(df_modeling ,"Modeling","updated_modeling_df","delete_modeling_col")
    # else:
    #     st.info(info["Modeling"][0])
  
   
    ########################################################################################################################################
    # Probability Threshold
    ########################################################################################################################################
    #repeated code
    with st.sidebar.header('1. Probability Threshold'):
        prob_treshold = st.sidebar.number_input('Enter the Probability Threshold', min_value=None, max_value=None, value=float(0.5))

    ########################################################################################################################################
    # Upload models
    ########################################################################################################################################
    # Upload File
    #repeated code
    with st.sidebar.header('2. Upload your model'):
        model_file = st.sidebar.file_uploader("Upload your model in PKL file", type=["pkl"], key="up_model_VS")
    
    with st.sidebar.expander("Whats a PKL file"):
        st.write(info["PKL"])
    
    st.sidebar.markdown("[Example PKL input file](%s)"% info["Links"][2])
    #repeated code
    def load_model(model):
        loaded_model = pickle.load(model)
        return loaded_model

    ########################################################################################################################################
    # Predictions
    ########################################################################################################################################
    def predict():
        
        #original_df=original_model
        to_delete = st.session_state.updated_modeling_df.columns
        #print(to_delete)
        deleted_col = original_model.drop(to_delete,axis=1)

        model = load_model(model_file)
        prediction = model.predict_proba(st.session_state.updated_modeling_df)
        # convert numpy array to pandas dataframe
        prediction = DataFrame(prediction, columns=['Prob_class_0', 'Prob_class_1'])
        # Join original data with predictions
        df_pred = deleted_col.join(prediction)

        #----------------------------------------------------------------------------------------------------------#
        # Binarize the outcomes
        df_pred['Pred_class'] = np.where(df_pred['Prob_class_1'] >= prob_treshold, 1, 0)

        #----------------------------------------------------------------------------------------------------------#
        # AD calculation
        distance_train_set = []
        distance_test_set = []
        AD = []
        k= int(round(pow((len(df_modeling)) ,1.0/3), 0))
        distance_train = getNeighborsDitance(df_modeling.to_numpy(), df_modeling.to_numpy() , k)
        distance_train_set.append(distance_train)
        distance_test = getNeighborsDitance(df_modeling.to_numpy(), df.to_numpy(), k)
        distance_test_set.append(distance_test)
        Dc = np.average(distance_train_set)-(0.5*np.std(distance_train_set))
        for i in range(len(df)):
            ad=0
            if distance_test_set[0][i] >= Dc:
                ad = 1
            AD.append(ad)

        AD = pd.DataFrame(AD)
        AD = AD.rename(columns = {0:'AD'})

        df_pred = df_pred.join(AD)
        return df_pred
    # Read Uploaded file and convert to pandas
    
    predicted=st.sidebar.button('Predict')
    
    if predicted:
        
        #----------------------------------------------------------------------------------------------------------#
        # Remove columns
        #df_pred = df_pred.drop(["descriptor"], axis=1)
        if model_file is not None:
            df_download = predict()
        else: st.info("Input your model first")
        
        # Table contains SMILES which we can convert to RDKit molecules (default name ROMol)
        """image=None
        if save_smiles:
            image=PandasTools.AddMoleculeColumnToFrame(smiles_column, smilesCol=name_smiles_vs, molCol='Image', includeFingerprints=False)"""
     
        # Print predictions
        st.header('**Predictions**')
        """st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:100px;border-radius:2%;">{df_download.to_html(escape=False)}</p>')
        if image is not None:
            st.markdown(image.to_html(escape=False), unsafe_allow_html=True)"""

        st.write(df_download)

    ########################################################################################################################################
    # Download files
    ########################################################################################################################################

        st.header('**Download files**')
        # File download
        filedownload(df_download)