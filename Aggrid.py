import pandas as pd 
import numpy as np
import glob
import os
from urllib.request import urlopen
import json
import streamlit as st 
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

st.set_page_config(page_title="Netflix Shows", layout="wide") 
st.title("Netlix shows analysis")

consulta_anno='2017','2018','2019','2020','2021','2022','2023','2024','2025'
@st.cache(allow_output_mutation=True)
def ReadApiINTFAccesos():
    resourceid = '540ea080-bf16-4d63-911f-3b4814e8e4f1'
    INTF_ACCESOS = pd.DataFrame()
    for anno in consulta_anno:
        consulta='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid + ''\
                 '&filters[anno]=' + anno + ''\
                 '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa&fields[]=id_departamento&fields[]=departamento&fields[]=id_municipio&fields[]=municipio'\
                 '&group_by=anno,trimestre,id_empresa,empresa,id_departamento,departamento,id_municipio,municipio'\
                 '&sum=accesos' 
        response_base = urlopen(consulta + '&limit=10000000') 
        json_base = json.loads(response_base.read())
        ACCESOS = pd.DataFrame(json_base['result']['records'])
        INTF_ACCESOS = INTF_ACCESOS.append(ACCESOS)
    INTF_ACCESOS.sum_accesos = INTF_ACCESOS.sum_accesos.astype('int64')
    INTF_ACCESOS = INTF_ACCESOS.rename(columns={'sum_accesos':'accesos'})
    return INTF_ACCESOS
InternetAccesos=ReadApiINTFAccesos()    

gb = GridOptionsBuilder.from_dataframe(InternetAccesos)

gb.configure_pagination()
gb.configure_side_bar()
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
gridOptions = gb.build()

AgGrid(InternetAccesos, gridOptions=gridOptions, enable_enterprise_modules=True)