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

import streamlit as st

import base64
import warnings
warnings.filterwarnings(action='ignore')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import pandas as pd

from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from st_aggrid import AgGrid
import utils
def app(df,s_state):

    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
    def persist_dataframe(updated_df,col_to_delete):
            # drop column from dataframe
            delete_col = st.session_state[col_to_delete]

            if delete_col in st.session_state[updated_df]:
                st.session_state[updated_df] = st.session_state[updated_df].drop(columns=[delete_col])
            else:
                st.sidebar.warning("Column previously deleted. Select another column.")
            with st.container():
                st.header("**Updated input data**") 
                AgGrid(st.session_state[updated_df])
                st.header('**Original input data**')
                AgGrid(df)

    def filedownload(df,data):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            st.header(f"**Download {data} data**")
            href = f'<a href="data:file/csv;base64,{b64}" download="{data}_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def remove_invalid(df):
        for i in df.index:
            try:
                smiles = df[name_smiles][i]
                m = Chem.MolFromSmiles(smiles)
            except:
                df.drop(i, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    ##################################################################
    # def remove_metals(df):
    #     badAtoms = Chem.MolFromSmarts('[!$([#1,#3,#11,#19,#4,#12,#20,#5,#6,#14,#7,#15,#8,#16,#9,#17,#35,#53])]')
    #     mols = []
    #     for i in df.index:
    #         smiles = df[name_smiles][i]
    #         m = Chem.MolFromSmiles(smiles,)
    #         if m.HasSubstructMatch(badAtoms):
    #             df.drop(i, inplace=True)
    #     df.reset_index(drop=True, inplace=True)
    #     return df
    # ##################################################################
    # def normalize_groups(df):
    #     mols = []
    #     for smi in df[name_smiles]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = rdMolStandardize.Normalize(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     norm = pd.DataFrame(mols, columns=["normalized_smiles"])
    #     df_normalized = df.join(norm)
    #     return df_normalized
    # ##################################################################
    # def neutralize(df):
    #     uncharger = rdMolStandardize.Uncharger()
    #     mols = []
    #     for smi in df['normalized_smiles']:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = uncharger.uncharge(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     neutral = pd.DataFrame(mols, columns=["neutralized_smiles"])
    #     df_neutral = df.join(neutral)
    #     return df_neutral
    # ##################################################################
    # def no_mixture(df):
    #     mols = []
    #     for smi in df["neutralized_smiles"]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = rdMolStandardize.FragmentParent(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     no_mixture = pd.DataFrame(mols, columns=["no_mixture_smiles"])
    #     df_no_mixture = df.join(no_mixture)
    #     return df_no_mixture
    # ##################################################################
    # def canonical_tautomer(df):
    #     te = rdMolStandardize.TautomerEnumerator()
    #     mols = []
    #     for smi in df["no_mixture_smiles"]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = te.Canonicalize(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     canonical_tautomer = pd.DataFrame(mols, columns=["canonical_tautomer"])
    #     df_canonical_tautomer = df.join(canonical_tautomer)
    #     return df_canonical_tautomer
    # ##################################################################
    # def smi_to_inchikey(df):
    #     inchi = []
    #     for smi in df["canonical_tautomer"]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = Chem.inchi.MolToInchiKey(m)
    #         inchi.append(m2)
    #     inchikey = pd.DataFrame(inchi, columns=["inchikey"])
    #     df_inchikey = df.join(inchikey)
    #     return df_inchikey
    # ##################################################################

    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################

    # Upload File
    
    #st.header('**Original input data**')

    # Read Uploaded file and convert to pandas
    if df is not None:
        # Read CSV data
        #df = pd.read_csv(uploaded_file, sep=',')
        curated_key = utils.Commons().CURATED_DF_KEY
    #custom = cur.Custom_Components()
    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
        cc = utils.Custom_Components()
        def persist_dataframe(updated_df,col_to_delete):
                # drop column from dataframe
                delete_col = st.session_state[col_to_delete]

                if delete_col in st.session_state[updated_df]:
                    st.session_state[updated_df] = st.session_state[updated_df].drop(columns=[delete_col])
                else:
                    st.sidebar.warning("Column previously deleted. Select another column.")
                with st.container():
                    st.header("**Updated input data**") 
                    AgGrid(st.session_state[updated_df])
                    st.header('**Original input data**')
                    AgGrid(df)

        def filedownload(df,data):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
                st.header(f"**Download {data} data**")
                href = f'<a href="data:file/csv;base64,{b64}" download="{data}_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        def remove_invalid(df,smiles_col):
            for i in df.index:
                try:
                    smiles = df[smiles_col][i]
                    m = Chem.MolFromSmiles(smiles)
                except:
                    df.drop(i, inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        # def smi_to_inchikey(df):
        #     inchi = []
        #     for smi in df["canonical_tautomer"]:
        #         m = Chem.MolFromSmiles(smi,sanitize=True,)
        #         m2 = Chem.inchi.MolToInchiKey(m)
        #         inchi.append(m2)
        #     inchikey = pd.DataFrame(inchi, columns=["inchikey"])
        #     df_inchikey = df.join(inchikey)
        #     return df_inchikey

        st.sidebar.write('---')

        # Select columns
        with st.sidebar.header('1. Enter column names'):
            name_smiles = st.sidebar.selectbox('Select column containing SMILES', options=df.columns, key="smiles_column")
            # name_activity = st.sidebar.selectbox(
            #     'Select column containing Activity (Active and Inactive should be 1 and 0, respectively or numerical values)', 
            #     options=df.columns, key="outcome_column"
            #     )
            curate = utils.Curation(name_smiles)
        ########################################################################################################################################
        # Sidebar - Select visual inspection
        ########################################################################################################################################

        st.sidebar.header('2. Visual inspection')

        st.sidebar.subheader('Select step for visual inspection')
                
        container = st.sidebar.container()
        _all = st.sidebar.checkbox("Select all")
        
        options=['Normalization',
                'Neutralization',
                'Mixture_removal',
                'Canonical_tautomers',
                'Chembl_Standardization',]
        if _all:
            selected_options = container.multiselect("Select one or more options:", options, options)
        else:
            selected_options =  container.multiselect("Select one or more options:", options)


        ########################################################################################################################################
        # Apply standardization
        ########################################################################################################################################

        if st.sidebar.button('Standardize'):

            #---------------------------------------------------------------------------------#
            # Remove invalid smiles
            remove_invalid(df,name_smiles)
            df[name_smiles] = curate.smiles_preparator(df[name_smiles])
            st.header("1. Invalid SMILES removed")
            cc.AgGrid(df,key = "invalid_smiles_removed")
            #---------------------------------------------------------------------------------#
            # Remove compounds with metals
            df = curate.remove_Salt_From_DF(df, name_smiles)
            df = curate.remove_metal(df, name_smiles)
            normalized = curate.normalize_groups(df)
            neutralized,_ = curate.neutralize(normalized,curate.curated_smiles)
            no_mixture,only_mixture = curate.remove_mixture(neutralized,curate.curated_smiles)
            canonical_tautomer,_ = curate.canonical_tautomer(no_mixture,curate.curated_smiles)
            standardized,_= curate.standardise(canonical_tautomer,curate.curated_smiles)
            #---------------------------------------------------------------------------------#
            # Normalize groups
            if options[0] in selected_options:
                cc.img_AgGrid(normalized,title="Normalized Groups",mol_col=name_smiles ,key="normalized_groups")        
            #----------------------------------------------------------------------------------#
            # Neutralize when possible
            if options[1] in selected_options:
 
                #st.header('**Neutralized Groups**')
                #if options[0] in selected_options:
                cc.img_AgGrid(neutralized,title="Neutralized Groups",mol_col=curate.curated_smiles,key="neutralized_groups")
            #---------------------------------------------------------------------------------#
            # Remove mixtures and salts
            if options[2] in selected_options:

                st.header('**Remove mixtures**')
                # if options[1] in selected_options:
                with st.container():
                    #st.header("Mixture")
                    if only_mixture=="No mixture":
                        st.write("**No mixture found**")
                    else:
                        cc.img_AgGrid(only_mixture,title = "Mixture",mol_col=curate.curated_smiles,key="mixture")
                    #st.header("No mixture")
                    cc.img_AgGrid(no_mixture,title = "No mixture",mol_col=curate.curated_smiles,key="no_mixture")
            #---------------------------------------------------------------------------------#
            #Generate canonical tautomers
            if options[3] in selected_options:
                cc.img_AgGrid(canonical_tautomer,title="Canonical tautomer",mol_col=curate.curated_smiles,key="canonical_tautomer")
            #---------------------------------------------------------------------------------#
            # Standardize using Chembl pipeline
            if options[4] in selected_options:
                cc.img_AgGrid(standardized,title="Chembl Standardization",mol_col=curate.curated_smiles,key="chembl_standardization")
            
            #standardized = curate.std_routine(canonical_tautomer,smiles = curate.curated_smiles)

            
        ########################################################################################################################################
        # Download Standardized with Duplicates
        ########################################################################################################################################
                
            # std_with_dup = canonical_tautomer.filter(items=["canonical_tautomer",])
            # std_with_dup.rename(columns={"canonical_tautomer": "SMILES",},inplace=True)
            # std_with_dup = std_with_dup.join(st.session_state.updated_df.drop(name_smiles, 1))

            filedownload(standardized,"Standardized with Duplicates")
        
        #--------------------------- Removal of duplicates------------------------------#

            # Generate InchiKey
            inchikey = curate.smi_to_inchikey(canonical_tautomer, 'curated_Smiles')

            no_dup = inchikey.drop_duplicates(subset='inchikey', keep="first")


        #--------------------------- Print dataframe without duplicates------------------------------#

           # Initialize session state if necessary
            if 'updated_df' not in st.session_state:
                st.session_state.updated_df = None  # Initialize with None or your default value

            # Assuming 'no_dup' is your processed DataFrame
            st.header('**Duplicates removed**')
            # Keep only curated smiles and outcome
            no_dup = no_dup.filter(items=["canonical_tautomer"])
            no_dup.rename(columns={"canonical_tautomer": "SMILES"}, inplace=True)

            # Check if 'updated_df' exists in session state and join if available
            if st.session_state.updated_df is not None:
                no_dup = no_dup.join(st.session_state.updated_df.drop(name_smiles, axis=1))

            # Display curated dataset
            st.write(no_dup)
        ########################################################################################################################################
        # Data download
        ########################################################################################################################################

            # File download
            filedownload(no_dup,"Curated")