import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import glob
import os
from urllib.request import urlopen
import json
from streamlit_folium import folium_static
from st_aggrid import AgGrid
import geopandas as gpd
import folium

##

## Funciones a utilizar
@st.cache
def flatten(t):
    return [item for sublist in t for item in sublist]
@st.cache    
def Cumulative(lists):
        cu_list = []
        length = len(lists)
        cu_list = [round(sum(lists[0:x:1]),5) for x in range(0, length+1)]
        return cu_list[1:]
@st.cache
def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')
@st.cache    
def f(dat, c='#ffffb3'):
    return [f'background-color: {c}' for i in dat]
def Average(list):
    return sum(list) / len(list)    

##Geojson
gdf = gpd.read_file("colombia2.geo.json")
geoJSON_states = list(gdf.NOMBRE_DPT.values)
denominations_json = []
Id_json = []
Colombian_DPTO=json.load(open("Colombia.geo.json", 'r'))
for index in range(len(Colombian_DPTO['features'])):
    denominations_json.append(Colombian_DPTO['features'][index]['properties']['NOMBRE_DPT'])
    Id_json.append(Colombian_DPTO['features'][index]['properties']['DPTO'])
denominations_json=sorted(denominations_json)
gdf=gdf.rename(columns={"NOMBRE_DPT":'departamento','DPTO':'id_departamento'})

##Definición funciones indicadores
@st.cache
def Participacion(df,column):
    part=df[column]/df[column].sum()
    return part
@st.cache    
def Stenbacka(df,column,gamma):
    if df.empresa.nunique()==1:
        Sten=1.0
    else:
        parti=sorted(df[column]/df[column].sum(),reverse=True)
        Highpart=parti[0:2]
        Sten=round(0.5*(1-gamma*(Highpart[0]**2-Highpart[1]**2)),4)
    return Sten
@st.cache
def IHH(df,column):
    part=(df[column]/df[column].sum())*100
    IHH=round(sum([elem**2 for elem in part]),2)
    return IHH   
@st.cache    
def Concentracion(df,column,periodo):
    df=df[df['periodo']==periodo]
    conclist=[];
    part=df[column]/df[column].sum()
    concentracion=np.cumsum(sorted(part,reverse=True)).tolist()
    concentracion.insert(0,periodo)
    conclist.append(concentracion)
    conc=pd.DataFrame.from_records(conclist).round(4)
    cols=[f'CR{i}' for i in range(1,len(conc.columns))]
    cols.insert(0,'periodo')
    conc.columns=cols
    return conc    
@st.cache 
def MediaEntropica(df,column):
    dfAgg=df.groupby(['empresa','municipio'])[column].sum().reset_index()
    dfAgg['TOTAL']=dfAgg[column].sum()
    dfAgg['SIJ']=dfAgg[column]/dfAgg['TOTAL']
    dfAgg['SI']=dfAgg['SIJ'].groupby(dfAgg['empresa']).transform('sum')
    dfAgg['WJ']=dfAgg['SIJ'].groupby(dfAgg['municipio']).transform('sum')
    dfAgg=dfAgg.sort_values(by='WJ',ascending=False)
    dfAgg['C1MED']=(dfAgg['SIJ']/dfAgg['WJ'])**((dfAgg['SIJ']/dfAgg['WJ']))
    dfAgg['C2MED']=dfAgg['C1MED'].groupby(dfAgg['municipio']).transform('prod')
    dfAgg['C3MED']=dfAgg['C2MED']**(dfAgg['WJ'])
    dfAgg['MED']=np.prod(np.array(dfAgg['C3MED'].unique().tolist()))
    dfAgg['C1MEE']=dfAgg['WJ']**dfAgg['WJ']
    dfAgg['MEE']=np.prod(np.array(dfAgg['C1MEE'].unique().tolist()))
    dfAgg['C1MEI']=(dfAgg['SI']/dfAgg['SIJ'])**((dfAgg['SIJ']/dfAgg['WJ']))
    dfAgg['C2MEI']=dfAgg['C1MEI'].groupby(dfAgg['municipio']).transform('prod')
    dfAgg['C3MEI']=dfAgg['C2MEI']**(dfAgg['WJ'])
    dfAgg['MEI']=np.prod(np.array(dfAgg['C3MEI'].unique().tolist()))
    dfAgg['Media entropica']=[a*b*c for a,b,c in zip(dfAgg['MED'].unique().tolist(),dfAgg['MEE'].unique().tolist(),dfAgg['MEI'].unique().tolist())][0]
#    dfAgg=dfAgg[dfAgg[column]>0]
    return dfAgg['Media entropica'].unique().tolist()[0],dfAgg
@st.cache 
def Linda(df,column,periodo):
    df=df[df['periodo']==periodo]
    part=sorted(df[column]/df[column].sum(),reverse=True)
    part=[x for x in part if x>1e-10]
    mm=[];
    lindlist=[];
    if df.empresa.nunique()==1:
        pass
    else:    
        for N in range(2,len(part)+1):
            xi=[];
            xNmi=[];
            N=N if N<len(part) else len(part)
            bla=part[0:N]
            for i in range(1,len(bla)):
                xi.append(Average(bla[:i]))
                xNmi.append(Average(bla[i:]))
            CocXi2=[x1/x2 for x1,x2 in zip(xi,xNmi)] 
            Lind2=round((1/(N*(N-1)))*(sum(CocXi2[:N-1])),5)   
            mm.append(Lind2)
    mm.insert(0,periodo)    
    lindlist.append(mm) 
    Linda=pd.DataFrame.from_records(lindlist).round(4)
    cols=[f'Linda ({i+1})' for i in range(1,len(Linda.columns))]
    cols.insert(0,'periodo')
    Linda.columns=cols    
    return Linda    
##
##Definición funciones para graficar los indicadores:
def PlotlyStenbacka(df):
    empresasdf=df['id_empresa'].unique().tolist()
    fig = make_subplots(rows=1, cols=1)
    dfStenbacka=df.groupby(['periodo'])['stenbacka'].mean().reset_index()
    for elem in empresasdf:
        fig.add_trace(go.Scatter(x=df[df['id_empresa']==elem]['periodo'],
        y=df[df['id_empresa']==elem]['participacion'],text=df[df['id_empresa']==elem]['empresa'],
        mode='lines+markers',line = dict(width=0.8),name='',hovertemplate =
        '<br><b>Empresa</b>:<br>'+'%{text}'+
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>Participación</b>: %{y:.4f}<br>')) 
    fig.add_trace(go.Scatter(x=dfStenbacka['periodo'],y=dfStenbacka['stenbacka'],name='',marker_color='rgba(128, 128, 128, 0.5)',fill='tozeroy',fillcolor='rgba(192, 192, 192, 0.15)',
        hovertemplate =
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>Stenbacka</b>: %{y:.4f}<br>'))    
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=None,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="Participación (%)", row=1, col=1)
    fig.update_layout(height=550,title="<b> Participación por periodo</b>",title_x=0.5,legend_title=None,font=dict(family="Helvetica",color=" black"))
    fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-90,showgrid=True, gridwidth=1, gridcolor='rgba(192, 192, 192, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(192, 192, 192, 0.4)')
    return fig

def PlotlyConcentracion(df):    
    fig = make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=flatten(df.iloc[:, [conc]].values),hovertemplate =
    '<br><b>Periodo</b>: %{x}<br>'+                         
    '<br><b>Concentración</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=None,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="Concentración", row=1, col=1)
    fig.update_layout(height=550,title="<b> Razón de concentración por periodo</b>",title_x=0.5,legend_title=None,font=dict(family="Helvetica",color=" black"))
    fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-90,showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.4)
    fig.add_hline(y=0.44, line_dash="dot",
              annotation_text="Baja", 
              annotation_position="bottom left")
    fig.add_hline(y=0.71, line_dash="dot",
              annotation_text="Alta", 
              annotation_position="top left",line_color="red")
    fig.add_hrect(
    y0=0.45, y1=0.699,
    fillcolor="orange", opacity=0.4,
    layer="below", line_width=0,row=1, col=1,annotation_text="Moderada",annotation_position="top left")
    return fig
def PlotlyPenetracion(df):
    fig = make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=df['penetracion'],
                         hovertemplate =
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>Penetración</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=None,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="Penetración", row=1, col=1)
    fig.update_layout(height=550,title="<b> Índice de penetración</b>",title_x=0.5,legend_title=None,font=dict(family="Helvetica",color=" black"))
    fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-90,showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_traces(marker_color='rgb(0,153,153)', marker_line_color='rgb(32,32,32)',
                      marker_line_width=1.5, opacity=0.4)
    return fig    

def PlotlyIHH(df):    
    fig = make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=df['IHH'],
                         hovertemplate =
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>IHH</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=None,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="Concentración", row=1, col=1)
    fig.update_layout(height=550,title="<b> Índice Herfindahl-Hirschman</b>",title_x=0.5,legend_title=None,font=dict(family="Helvetica",color=" black"))
    fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-90,showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_traces(marker_color='rgb(255,0,0)', marker_line_color='rgb(204,0,0)',
                      marker_line_width=1.5, opacity=0.4)
    fig.add_hline(y=1500, line_dash="dot",
                  annotation_text="No concentrado", 
                  annotation_position="bottom left")
    fig.add_hline(y=2500, line_dash="dot",
                  annotation_text="Altamente concentrado", 
                  annotation_position="top left",line_color="red")
    fig.add_hrect(
        y0=1501, y1=2499,
        fillcolor="rgb(0,0,102)", opacity=0.6,
        layer="below", line_width=0,row=1, col=1,annotation_text="Concentrado",annotation_position="bottom left")
    return fig    

def PlotlyMEntropica(df):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=df['periodo'],
         y=df['media entropica'],
        name='',hovertemplate =
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>MEDIA ENTROPICA</b>: %{y:.4f}<br>')) 
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=None,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="MEDIA ENTROPICA", row=1, col=1)
    fig.update_layout(height=550,title="<b>Evolución Media entrópica</b>",title_x=0.5,legend_title=None,font=dict(family="Helvetica",color=" black"))
    fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-90,showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_traces(marker_color='rgb(0,153,0)', marker_line_color='rgb(25,51,0)',
                      marker_line_width=1.5, opacity=0.5)
    return fig

def PlotlyMentropicaTorta(df):
    fig = px.pie(df, values='WJ', names='municipio', color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=20, uniformtext_mode='hide',height=500, width=280)
    fig.update_traces(hoverinfo='label+percent', textinfo='value',
                  marker=dict(line=dict(color='#000000', width=1)))
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),yaxis = dict(domain=(0.1,1)))    
    fig.update_layout(legend=dict(
    orientation='h',
    y=0,
    x=0.1))
    return fig

def PlotlyLinda(df):    
    fig = make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=flatten(df.iloc[:, [lind-1]].values),hovertemplate =
    '<br><b>Periodo</b>: %{x}<br>'+                         
    '<br><b>Linda</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=None,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="Linda", row=1, col=1)
    fig.update_layout(height=550,title="<b> Índice de Linda por periodo</b>",title_x=0.5,legend_title=None,font=dict(family="Helvetica",color=" black"))
    fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-90,showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)',type="log", tickvals=[0.5,0.7,0.8,0.9,1.0,1.5,2.0,3.0,5.0,10,50,100,250,500,750,1000])
    fig.update_traces(marker_color='rgb(127,0,255)', marker_line_color='rgb(51,0,102)',
                  marker_line_width=1.5, opacity=0.4)
    return fig
def PlotlyLinda2(df):
    fig= make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=df['Linda (2)'],hovertemplate =
    '<br><b>Periodo</b>: %{x}<br>'+                         
    '<br><b>Linda</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=None,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="Linda", row=1, col=1)
    fig.update_layout(height=550,title="<b> Índice de Linda por periodo</b>",title_x=0.5,legend_title=None,font=dict(family="Helvetica",color=" black"))
    fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-90,showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)',type="log", tickvals=[0.5,0.7,0.8,0.9,1.0,1.5,2.0,3.0,5.0,10,50,100,250,500,750,1000])
    fig.update_traces(marker_color='rgb(127,0,255)', marker_line_color='rgb(51,0,102)',
                  marker_line_width=1.5, opacity=0.4)        
    return fig                
LogoComision="data:image/png;base64,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"
LogoComision2="https://www.postdata.gov.co/sites/all/themes/nuboot_radix/logo-crc-blanco.png"
LogoMercadoTIC="https://upload.wikimedia.org/wikipedia/commons/4/41/Noun_project_network_icon_1365244_cc.svg"
# Set page title and favicon.

st.set_page_config(
    page_title="Batería de indicadores", page_icon=LogoComision,layout="wide",initial_sidebar_state="expanded")


st.markdown("""<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">""",unsafe_allow_html=True)     
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: -250px;
    }
    </style>
    """,
    unsafe_allow_html=True)   
st.markdown("""<style type="text/css">
    h1{ background: #ffde00;
    text-align: center;
    padding: 15px;
    font-family: sans-serif;
    font-size:1.60rem;
    color: black;
    position:fixed;
    width:100%;
    z-index:9999;
    top:80px;
    left:0;}
    .barra-superior{top: 0;
    position: fixed;
    background-color: #27348b;
    width: 100%;
    color:white;
    z-index: 999;
    height: 80px;
    left: 0px;
    text-align: center;
    padding: 0px;
    font-size: 36px;
    font-weight: 700;
    }
    .main, .css-1lcbmhc > div{margin-top:135px;}
    .css-y3whyl, .css-xqnn38 {background-color:#ccc}
    .css-1uvyptr:hover,.css-1uvyptr {background: #ccc}
    .block-container {padding-top:0;}
    h2{
    background: #fffdf7;
    text-align: center;
    padding: 10px;
    text-decoration: underline;
    text-decoration-style: double;
    color: #27348b;}
    h3{ border-bottom: 2px solid #27348b;
    border-left: 10px solid #27348b;
    background: #fffdf7;
    padding: 10px;
    color: black;}
    .imagen-flotar{float:left;}
    @media (max-width:1230px){
        .barra-superior{height:160px;} 
        .main, .css-1lcbmhc > div{margin-top:215px;}
        .imagen-flotar{float:none}
        h1{top:160px;}
    }    
    </style>""", unsafe_allow_html=True)  
st.markdown("""
<div class="barra-superior">
    <div class="imagen-flotar" style="height: 70px; left: 10px; padding:15px">
        <a class="imagen-flotar" style="float:left;" href="https://www.crcom.gov.co" title="CRC">
            <img src="https://www.postdata.gov.co/sites/all/themes/nuboot_radix/logo-crc-blanco.png" alt="CRC" style="height:40px">
        </a>
        <a class="imagen-flotar" style="padding-left:10px;" href="https://www.postdata.gov.co" title="Postdata">
            <img src="https://www.postdata.gov.co/sites/default/files/postdata-logo.png" alt="Inicio" style="height:40px">
        </a>
    </div>
    <div class="imagen-flotar" style="height: 80px; left: 300px; padding:5px">
        <a class="imagen-flotar" href="https://www.crcom.gov.co" title="CRC">
            <img src="https://www.postdata.gov.co/sites/default/files/bateria-indicadores.png" alt="CRC" style="">
        </a>
    </div>
</div>""",unsafe_allow_html=True)
#st.sidebar.image(LogoComision2, use_column_width=True)
#st.sidebar.image(LogoMercadoTIC, width=100)
st.markdown(r""" **<center><ins>Guía de uso de la batería de indicadores para el análisis de competencia</ins></center>**
- Use el menú de la barra de la izquierda para seleccionar el mercado sobre el cuál le gustaría realizar el cálculo de los indicadores.
- Elija el ámbito del mercado: Departamental, Municipal, Nacional.
- Escoja el indicador a calcular.
- Dependiendo del ámbito y el indicador, interactúe con los parámetros establecidos, tal como periodo, municipio, número de empresas, etc.
""",unsafe_allow_html=True)  
st.sidebar.markdown("""<b>Seleccione el indicador a calcular</b>""", unsafe_allow_html=True)
@st.cache(allow_output_mutation=True)
def loadCorreo():
    data=pd.read_csv("https://raw.githubusercontent.com/sbermudezf/ComisionRegulacionComunicaciones/main/ING_ENV_CORREO_IC.csv",delimiter=';',error_bad_lines=False,encoding='latin1')
    return data
@st.cache(allow_output_mutation=True)  
def loadMExpresa():
    data=pd.read_csv("https://raw.githubusercontent.com/sbermudezf/ComisionRegulacionComunicaciones/main/ING_ENV_ME_IC.csv",delimiter=';',error_bad_lines=False,encoding='latin1')
    return data
Correo=loadCorreo()
MenExp=loadMExpresa()
MenExp.NUMERO_TOTAL_ENVIOS=MenExp.NUMERO_TOTAL_ENVIOS.astype(str).str.replace(',','.').astype('float').astype('int64')
MenExp.INGRESOS=MenExp.INGRESOS.astype(str).str.replace(',','.').astype('float')
Correo.INGRESOS=Correo.INGRESOS.astype(str).str.replace(',','.').astype('float')

Postales=pd.concat([Correo,MenExp])
Postales.columns=[x.lower() for x in Postales.columns]
Postales.insert(2,'periodo',Postales.anno.astype('str')+'-T'+Postales.trimestre.astype('str'))


##NÚMERO DE PERSONAS
Personas=pd.read_csv("https://raw.githubusercontent.com/sbermudezf/ComisionRegulacionComunicaciones/main/POBLACION.csv",delimiter=';')
Personas.columns=[x.lower() for x in Personas.columns]
Personas.id_municipio=Personas.id_municipio.astype(str)
Personas.id_departamento=Personas.id_departamento.astype(str)
Personas.anno=Personas.anno.astype('str')

select_ambito = st.sidebar.selectbox('Seleccionar ámbito de aplicación',
                                    ['Nacional','Internacional'])
                                    
st.title("Mercado postal")

if select_ambito =='Nacional':
    nacional=Postales[Postales['ambito'].isin(['Local','Nacional'])]
    select_envio = st.sidebar.selectbox('Seleccionar envío',['Individual','Masivo'])
    if select_envio== 'Individual':
        Individual=nacional[nacional['tipo_envio']=='Envíos Individuales']
        col1, col2 ,col3= st.columns(3)
        with col1:
            select_objeto=st.selectbox('Seleccionar tipo de objeto',['Documentos','Paquetes'])
        with col2:    
            select_dimension = st.selectbox('Seleccione ámbito aplicación',['Nacional','Municipal'])
        with col3:
            select_variable = st.selectbox('Seleccione la variable',['Envíos','Ingresos'])
            
        if select_objeto=='Documentos':
            dfIngresos=[];dfIngresos2=[];dfIngresos3=[];
            dfEnvios=[];dfEnvios2=[];dfEnvios3=[];
            Documentos=Individual[Individual['tipo_objeto']=='Documentos']
            Documentos.drop(['anno','trimestre','id_tipo_envio','tipo_envio','id_tipo_objeto','id_ambito'],axis=1, inplace=True)
            PERIODOS=['2020-T3','2020-T4','2021-T1','2021-T2']
            DocumentosnacIng=Documentos.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
            DocumentosnacEnv=Documentos.groupby(['periodo','empresa','id_empresa'])['numero_total_envios'].sum().reset_index()
            
            with st.expander('Datos documentos'):
                AgGrid(Documentos)
            if select_dimension == 'Nacional':            
                select_indicador = st.sidebar.selectbox('Indicador',['Stenbacka', 'Concentración','IHH','Linda','Penetración'])
            ## Información sobre los indicadores
                if select_indicador == 'Stenbacka':
                    st.write("### Índice de Stenbacka")
                    st.markdown("Este índice de dominancia es una medida para identificar cuándo una empresa podría tener posición dominante en un mercado determinado. Se considera la participación de mercado de las dos empresas con mayor participación y se calcula un umbral de cuota de mercado después del cual la empresa lider posiblemente ostentaría posición de dominio. Cualquier couta de mercado superior a dicho umbral podría significar una dominancia en el mercado.")
                    #st.latex(r'''S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]''')       
                    with st.expander("Información adicional índice de Stenbacka"):
                        st.write(r""" El índice de Stenbacka está dado por la siguiente ecuación""")
                        st.latex(r"""S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]""")
                        st.write(r"""
    **Donde**
    -   $S^{2}_{1}$ y $S^{2}_{2}$ Corresponden a las participaciones de mercado de las dos empresas más grandes, respectivamente.
    -   $\gamma$ es un parámetro de competencia que puede incluir aspectos como: existencia de compradores con poder de mercado, regulación económica, presencia de derechos de propiedad, barreras a la entrada, entre otros (Lis-Guitiérrez, 2013).                
                    """,unsafe_allow_html=True)
                if select_indicador == 'Concentración':
                    st.write("### Razón de concentración")
                    st.markdown("La razón de concentración es un índice que mide las participaciones acumuladas de las empresas lideres en el mercado. Toma valores entre 0 y 1.")            
                    with st.expander("Información adicional razón de concentración"):
                        st.write("La concentración se calcula de la siguiente forma:")
                        st.latex(r''' CR_{n}=S_1+S_2+S_3+...+S_n=\sum_{i=1}^{n}S_{i}''')
                        st.write(r""" **Donde**:
    -   $S_{i}$ es la participación de mercado de la i-ésima empresa.
    -   $n$ es el número total de empresas consideradas.

    De acuerdo con Stazhkova, Kotcofana & Protasov (2017), para un $n = 3$ se pueden considerar los siguientes rangos de concentración para un mercado:

    | Concetración | Rango         |
    |--------------|---------------|
    | Baja         | $<0,45$       |
    | Moderada     | $0,45 - 0,70$ |
    | Alta         | $>0,70$       |
                    
                    
    """)
                if select_indicador == 'IHH':
                    st.write("### Índice de Herfindahl-Hirschman")
                    st.markdown("El IHH es el índice más aceptado como medida de concentración de la oferta en un mercado. Su cálculo se expresa como la suma de los cuadrados de las participaciones de las empresas que componen el mercado. El índice máximo se obtiene para un monopolio y corresponde a 10000.")            
                    with st.expander("Información adicional IHH"):
                        st.write("La fórmula del IHH está dada como")
                        st.latex(r'''IHH=\sum_{i=1}^{n}S_{i}^{2}''')
                        st.write(r"""**Donde:**
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    De acuerdo con el Departamento de Justicia y la Comisión Federal de Comercio de Estados Unidos (2010), se puede categorizar a un mercado de acuerdo a los siguientes rangos de este índice:

    | Mercado                   | Rango          |
    |---------------------------|----------------|
    | Muy competitivo           | $<100$         |
    | Desconcentrado            | $100 - 1500$   |
    | Moderadamente concentrado | $>1500 - 2500$ |
    | Altamente concentrado     | $>2500$        |                
                    """)
                if select_indicador == 'Linda':         
                    st.write("### Índice de Linda")               
                    st.markdown("Este índice es utilizado para medir la desigualdad entre diferentes cuotas de mercado e identificar posibles oligopolios. El índice tomará valores cercanos a 1 en la medida que la participación en el mercado del grupo de empresas grandes es mayor que la participación del grupo de empresas pequeñas.")                    
                    with st.expander("Información adicional indicador de linda"): 
                        st.write("El indicador de Linda está dado por la siguiente ecuación:")
                        st.latex(r'''L = \frac{1}{N(N-1)} \sum_{i=1}^{N-1} (\frac{\overline{X}_{i}}{\overline{X}_{N-i}})''')
                        st.write(r"""**Donde**:
    - $\overline{X}_{i}$ es la participación de mercado media de las primeras i-ésimas empresas.
    - $\overline{X}_{N-i}$ es la partipación de mercado media de las i-ésimas empresas restantes.

    De acuerdo con Martinez (2017), se pueden considerar los siguientes rangos de concentración para un mercado:

    | Concentración   | Rango         |
    |-----------------|---------------|
    | Baja            | $<0,20$       |
    | Moderada        | $0,20 - 0,50$ |
    | Concentrada     | $>0,50 - 1$   |
    | Alta            | $>1$          |""",unsafe_allow_html=True)
                if select_indicador == 'Penetración':
                    st.write("### Índice de penetración")
                    st.markdown("El índice de penetración es usado para...")    

            ## Cálculo de los indicadores
                if select_indicador == 'Stenbacka':
                    gamma=st.slider('Seleccionar valor gamma',0.0,2.0,0.1)
                    for elem in PERIODOS:
                        prIn=DocumentosnacIng[DocumentosnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
                        prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
                        dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))

                        prEn=DocumentosnacEnv[DocumentosnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',Participacion(prEn,'numero_total_envios'))
                        prEn.insert(4,'stenbacka',Stenbacka(prEn,'numero_total_envios',gamma))
                        dfEnvios.append(prEn.sort_values(by='participacion',ascending=False))                        
                        
                    InggroupPart=pd.concat(dfIngresos)
                    InggroupPart.participacion=InggroupPart.participacion.round(5)
                    EnvgroupPart=pd.concat(dfEnvios)
                    EnvgroupPart.participacion=EnvgroupPart.participacion.round(5)
                    if select_variable == 'Ingresos':
                        AgGrid(InggroupPart)
                        fig1=PlotlyStenbacka(InggroupPart)
                        st.plotly_chart(fig1, use_container_width=True)
                    if select_variable == 'Envíos':
                        AgGrid(EnvgroupPart)
                        fig2=PlotlyStenbacka(EnvgroupPart)
                        st.plotly_chart(fig2, use_container_width=True)                        

                if select_indicador == 'Concentración':
                    dflistEnv=[];dflistIng=[]
                    
                    for elem in PERIODOS:
                        dflistEnv.append(Concentracion(DocumentosnacEnv,'numero_total_envios',elem))
                        dflistIng.append(Concentracion(DocumentosnacIng,'ingresos',elem))
                    ConcEnv=pd.concat(dflistEnv).fillna(1.0)
                    ConcIng=pd.concat(dflistIng).fillna(1.0)
                                             
                    if select_variable == "Envíos":
                        colsconEnv=ConcEnv.columns.values.tolist()
                        conc=st.slider('Seleccionar el número de empresas',1,len(colsconEnv)-1,1,1)
                        fig4=PlotlyConcentracion(ConcEnv)
                        st.write(ConcEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconEnv[conc]]))
                        st.plotly_chart(fig4,use_container_width=True)
                    if select_variable == "Ingresos":
                        colsconIng=ConcIng.columns.values.tolist()
                        conc=st.slider('Seleccione el número de empresas',1,len(colsconIng)-1,1,1)
                        fig5=PlotlyConcentracion(ConcIng)
                        st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                        st.plotly_chart(fig5,use_container_width=True)                    

                if select_indicador == 'IHH':
                    for elem in PERIODOS:
                        prEn=DocumentosnacEnv[DocumentosnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',(prEn['numero_total_envios']/prEn['numero_total_envios'].sum())*100)
                        prEn.insert(4,'IHH',IHH(prEn,'numero_total_envios'))
                        dfEnvios3.append(prEn.sort_values(by='participacion',ascending=False))
                        ##
                        prIn=DocumentosnacIng[DocumentosnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                        prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                        dfIngresos3.append(prIn.sort_values(by='participacion',ascending=False))
                        ##

                    EnvgroupPart3=pd.concat(dfEnvios3)
                    InggroupPart3=pd.concat(dfIngresos3)
                    
                    IHHEnv=EnvgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
                    IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()                
                    
                    ##Gráficas
                    
                    fig7 = PlotlyIHH(IHHEnv)   
                    fig8 = PlotlyIHH(IHHIng)  
                    
                    if select_variable == "Envíos":
                        AgGrid(EnvgroupPart3)
                        st.plotly_chart(fig7,use_container_width=True)
                    if select_variable == "Ingresos":
                        AgGrid(InggroupPart3)
                        st.plotly_chart(fig8,use_container_width=True)

                if select_indicador == 'Linda':
                    dflistEnv2=[];dflistIng2=[]                    
                    for elem in PERIODOS:
                        dflistEnv2.append(Linda(DocumentosnacEnv,'numero_total_envios',elem))
                        dflistIng2.append(Linda(DocumentosnacIng,'ingresos',elem))
                    LindEnv=pd.concat(dflistEnv2).reset_index().drop('index',axis=1).fillna(np.nan)
                    LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
         
                    if select_variable == "Envíos":
                        LindconEnv=LindEnv.columns.values.tolist()
                        lind=st.slider('Seleccionar nivel',2,len(LindconEnv),2,1)
                        fig10=PlotlyLinda(LindEnv)
                        st.write(LindEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconEnv[lind-1]]))
                        st.plotly_chart(fig10,use_container_width=True)
                    if select_variable == "Ingresos":
                        LindconIng=LindIng.columns.values.tolist()            
                        lind=st.slider('Seleccionar nivel',2,len(LindconIng),2,1)
                        fig11=PlotlyLinda(LindIng)
                        st.write(LindIng.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconIng[lind-1]]))
                        st.plotly_chart(fig11,use_container_width=True)

                if select_indicador == 'Penetración':
                    PersonasNac=Personas.groupby(['anno'])['poblacion'].sum()  
                    EnvNac=DocumentosnacEnv.groupby(['periodo'])['numero_total_envios'].sum().reset_index()
                    EnvNac.insert(0,'anno',EnvNac.periodo.str.split('-',expand=True)[0])
                    PenetracionNac=EnvNac.merge(PersonasNac, on=['anno'], how='left')
                    PenetracionNac.insert(4,'penetracion',PenetracionNac['numero_total_envios']/PenetracionNac['poblacion'])
                    PenetracionNac.penetracion=PenetracionNac.penetracion.round(3)
                    if select_variable=='Envíos':
                        fig12=PlotlyPenetracion(PenetracionNac)
                        AgGrid(PenetracionNac[['periodo','numero_total_envios','poblacion','penetracion']])
                        st.plotly_chart(fig12,use_container_width=True)
                    if select_variable=='Ingresos':
                        st.write("El indicador de penetración sólo está definido para la variable de Envíos.")   
                                                
        if select_objeto=='Paquetes':
            dfIngresos=[];dfIngresos2=[];dfIngresos3=[];
            dfEnvios=[];dfEnvios2=[];dfEnvios3=[];        
            Paquetes=Individual[Individual['tipo_objeto']=='Paquetes']
            Paquetes.drop(['anno','trimestre','id_tipo_envio','tipo_envio','id_tipo_objeto','id_ambito'],axis=1, inplace=True)
            with st.expander('Datos paquetes'):
                AgGrid(Paquetes)
            PESO = st.select_slider('Seleccione rango de peso',Paquetes.rango_peso_envio.unique().tolist())   
            PERIODOS=['2020-T3','2020-T4','2021-T1','2021-T2']            
            PaquetesPeso=Paquetes[Paquetes['rango_peso_envio']==PESO]
            PaquetesnacIng=PaquetesPeso.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
            PaquetesnacEnv=PaquetesPeso.groupby(['periodo','empresa','id_empresa'])['numero_total_envios'].sum().reset_index()
            
            if select_dimension == 'Nacional':       
                select_indicador = st.sidebar.selectbox('Indicador',['Stenbacka', 'Concentración','IHH','Linda','Penetración'])
                ## Información sobre los indicadores
                if select_indicador == 'Stenbacka':
                    st.write("### Índice de Stenbacka")
                    st.markdown("Este índice de dominancia es una medida para identificar cuándo una empresa podría tener posición dominante en un mercado determinado. Se considera la participación de mercado de las dos empresas con mayor participación y se calcula un umbral de cuota de mercado después del cual la empresa lider posiblemente ostentaría posición de dominio. Cualquier couta de mercado superior a dicho umbral podría significar una dominancia en el mercado.")
                    #st.latex(r'''S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]''')       
                    with st.expander("Información adicional índice de Stenbacka"):
                        st.write(r""" El índice de Stenbacka está dado por la siguiente ecuación""")
                        st.latex(r"""S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]""")
                        st.write(r"""
**Donde**
-   $S^{2}_{1}$ y $S^{2}_{2}$ Corresponden a las participaciones de mercado de las dos empresas más grandes, respectivamente.
-   $\gamma$ es un parámetro de competencia que puede incluir aspectos como: existencia de compradores con poder de mercado, regulación económica, presencia de derechos de propiedad, barreras a la entrada, entre otros (Lis-Guitiérrez, 2013).                
                """,unsafe_allow_html=True)
                if select_indicador == 'Concentración':
                    st.write("### Razón de concentración")
                    st.markdown("La razón de concentración es un índice que mide las participaciones acumuladas de las empresas lideres en el mercado. Toma valores entre 0 y 1.")            
                    with st.expander("Información adicional razón de concentración"):
                        st.write("La concentración se calcula de la siguiente forma:")
                        st.latex(r''' CR_{n}=S_1+S_2+S_3+...+S_n=\sum_{i=1}^{n}S_{i}''')
                        st.write(r""" **Donde**:
-   $S_{i}$ es la participación de mercado de la i-ésima empresa.
-   $n$ es el número total de empresas consideradas.

De acuerdo con Stazhkova, Kotcofana & Protasov (2017), para un $n = 3$ se pueden considerar los siguientes rangos de concentración para un mercado:

| Concetración | Rango         |
|--------------|---------------|
| Baja         | $<0,45$       |
| Moderada     | $0,45 - 0,70$ |
| Alta         | $>0,70$       |
                
                
""")
                if select_indicador == 'IHH':
                    st.write("### Índice de Herfindahl-Hirschman")
                    st.markdown("El IHH es el índice más aceptado como medida de concentración de la oferta en un mercado. Su cálculo se expresa como la suma de los cuadrados de las participaciones de las empresas que componen el mercado. El índice máximo se obtiene para un monopolio y corresponde a 10000.")            
                    with st.expander("Información adicional IHH"):
                        st.write("La fórmula del IHH está dada como")
                        st.latex(r'''IHH=\sum_{i=1}^{n}S_{i}^{2}''')
                        st.write(r"""**Donde:**
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    De acuerdo con el Departamento de Justicia y la Comisión Federal de Comercio de Estados Unidos (2010), se puede categorizar a un mercado de acuerdo a los siguientes rangos de este índice:

    | Mercado                   | Rango          |
    |---------------------------|----------------|
    | Muy competitivo           | $<100$         |
    | Desconcentrado            | $100 - 1500$   |
    | Moderadamente concentrado | $>1500 - 2500$ |
    | Altamente concentrado     | $>2500$        |                
                    """)
                if select_indicador == 'Linda':
                    st.write("### Índice de Linda")               
                    st.markdown("Este índice es utilizado para medir la desigualdad entre diferentes cuotas de mercado e identificar posibles oligopolios. El índice tomará valores cercanos a 1 en la medida que la participación en el mercado del grupo de empresas grandes es mayor que la participación del grupo de empresas pequeñas.")                    
                    with st.expander("Información adicional indicador de linda"): 
                        st.write("El indicador de Linda está dado por la siguiente ecuación:")
                        st.latex(r'''L = \frac{1}{N(N-1)} \sum_{i=1}^{N-1} (\frac{\overline{X}_{i}}{\overline{X}_{N-i}})''')
                        st.write(r"""**Donde**:
- $\overline{X}_{i}$ es la participación de mercado media de las primeras i-ésimas empresas.
- $\overline{X}_{N-i}$ es la partipación de mercado media de las i-ésimas empresas restantes.

De acuerdo con Martinez (2017), se pueden considerar los siguientes rangos de concentración para un mercado:

| Concentración   | Rango         |
|-----------------|---------------|
| Baja            | $<0,20$       |
| Moderada        | $0,20 - 0,50$ |
| Concentrada     | $>0,50 - 1$   |
| Alta            | $>1$          |""",unsafe_allow_html=True)
                if select_indicador == 'Penetración':
                    st.write("### Índice de penetración")
                    st.markdown("El índice de penetración es usado para...")    

                ## Cálculo de los indicadores
                if select_indicador == 'Stenbacka':
                    gamma=st.slider('Seleccionar valor gamma',0.0,2.0,0.1)
                    for elem in PERIODOS:
                        prIn=PaquetesnacIng[PaquetesnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
                        prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
                        dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))

                        prEn=PaquetesnacEnv[PaquetesnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',Participacion(prEn,'numero_total_envios'))
                        prEn.insert(4,'stenbacka',Stenbacka(prEn,'numero_total_envios',gamma))
                        dfEnvios.append(prEn.sort_values(by='participacion',ascending=False))                        
                        
                    InggroupPart=pd.concat(dfIngresos)
                    InggroupPart.participacion=InggroupPart.participacion.round(5)
                    EnvgroupPart=pd.concat(dfEnvios)
                    EnvgroupPart.participacion=EnvgroupPart.participacion.round(5)
                    if select_variable == 'Ingresos':
                        AgGrid(InggroupPart)
                        fig1=PlotlyStenbacka(InggroupPart)
                        st.plotly_chart(fig1, use_container_width=True)
                    if select_variable == 'Envíos':
                        AgGrid(EnvgroupPart)
                        fig2=PlotlyStenbacka(EnvgroupPart)
                        st.plotly_chart(fig2, use_container_width=True)  

                if select_indicador == 'Concentración':
                    dflistEnv=[];dflistIng=[]
                    
                    for elem in PERIODOS:
                        dflistEnv.append(Concentracion(PaquetesnacEnv,'numero_total_envios',elem))
                        dflistIng.append(Concentracion(PaquetesnacIng,'ingresos',elem))
                    ConcEnv=pd.concat(dflistEnv).fillna(1.0)
                    ConcIng=pd.concat(dflistIng).fillna(1.0)
                                             
                    if select_variable == "Envíos":
                        colsconEnv=ConcEnv.columns.values.tolist()
                        value1= len(colsconEnv)-1 if len(colsconEnv)-1 >1 else 2
                        conc=st.slider('Seleccionar el número de empresas',1,value1,1,1)
                        fig4=PlotlyConcentracion(ConcEnv)
                        st.write(ConcEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconEnv[conc]]))
                        st.plotly_chart(fig4,use_container_width=True)
                    if select_variable == "Ingresos":
                        colsconIng=ConcIng.columns.values.tolist()
                        value1= len(colsconIng)-1 if len(colsconIng)-1 >1 else 2
                        conc=st.slider('Seleccione el número de empresas',1,value1,1,1)
                        fig5=PlotlyConcentracion(ConcIng)
                        st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                        st.plotly_chart(fig5,use_container_width=True)                    

                if select_indicador == 'IHH':
                    for elem in PERIODOS:
                        prEn=PaquetesnacEnv[PaquetesnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',(prEn['numero_total_envios']/prEn['numero_total_envios'].sum())*100)
                        prEn.insert(4,'IHH',IHH(prEn,'numero_total_envios'))
                        dfEnvios3.append(prEn.sort_values(by='participacion',ascending=False))
                        ##
                        prIn=PaquetesnacIng[PaquetesnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                        prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                        dfIngresos3.append(prIn.sort_values(by='participacion',ascending=False))
                        ##

                    EnvgroupPart3=pd.concat(dfEnvios3)
                    InggroupPart3=pd.concat(dfIngresos3)
                    
                    IHHEnv=EnvgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
                    IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()                
                    
                    ##Gráficas
                    
                    fig7 = PlotlyIHH(IHHEnv)   
                    fig8 = PlotlyIHH(IHHIng)  
                    
                    if select_variable == "Envíos":
                        AgGrid(EnvgroupPart3)
                        st.plotly_chart(fig7,use_container_width=True)
                    if select_variable == "Ingresos":
                        AgGrid(InggroupPart3)
                        st.plotly_chart(fig8,use_container_width=True)

                if select_indicador == 'Linda':
                    dflistEnv2=[];dflistIng2=[];nempresaIng=[]; nempresaEnv=[];                     
                    for elem in PERIODOS:
                        prEnv=PaquetesnacEnv[PaquetesnacEnv['periodo']==elem]
                        nempresaEnv.append(prEnv.empresa.nunique())
                        prIng=PaquetesnacIng[PaquetesnacIng['periodo']==elem]
                        nempresaIng.append(prIng.empresa.nunique())                    
                        dflistEnv2.append(Linda(PaquetesnacEnv,'numero_total_envios',elem))
                        dflistIng2.append(Linda(PaquetesnacIng,'ingresos',elem))
                    NemphisEnv=max(nempresaIng)
                    NemphisIng=max(nempresaEnv)    
                    LindEnv=pd.concat(dflistEnv2).reset_index().drop('index',axis=1).fillna(np.nan)
                    LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
         
                    if select_variable == "Envíos":
                        LindconEnv=LindEnv.columns.values.tolist()
                        if NemphisEnv==1:
                            st.write("El índice de linda no está definido pues sólo hay una empresa")  
                        else:            
                            lind=st.slider('Seleccionar nivel',2,len(LindconEnv),2,1)
                            fig10=PlotlyLinda(LindEnv)
                            st.write(LindEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconEnv[lind-1]]))
                            st.plotly_chart(fig10,use_container_width=True)
                    if select_variable == "Ingresos":                    
                        LindconIng=LindIng.columns.values.tolist()        
                        if NemphisIng==1:
                            st.write("El índice de linda no está definido pues sólo hay una empresa")  
                        else:            
                            lind=st.slider('Seleccionar nivel',2,len(LindconIng),2,1)
                            fig11=PlotlyLinda(LindIng)
                            st.write(LindIng.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconIng[lind-1]]))
                            st.plotly_chart(fig11,use_container_width=True)

                if select_indicador == 'Penetración':
                    PersonasNac=Personas.groupby(['anno'])['poblacion'].sum()  
                    EnvNac=PaquetesnacEnv.groupby(['periodo'])['numero_total_envios'].sum().reset_index()
                    EnvNac.insert(0,'anno',EnvNac.periodo.str.split('-',expand=True)[0])
                    PenetracionNac=EnvNac.merge(PersonasNac, on=['anno'], how='left')
                    PenetracionNac.insert(4,'penetracion',PenetracionNac['numero_total_envios']/PenetracionNac['poblacion'])
                    PenetracionNac.penetracion=PenetracionNac.penetracion.round(3)
                    
                    if select_variable=='Envíos':
                        fig12=PlotlyPenetracion(PenetracionNac)
                        AgGrid(PenetracionNac[['periodo','numero_total_envios','poblacion','penetracion']])
                        st.plotly_chart(fig12,use_container_width=True)
                    if select_variable=='Ingresos':
                        st.write("El indicador de penetración sólo está definido para la variable de Envíos.")   
                       
    if select_envio== 'Masivo':
        Masivo=nacional[nacional['tipo_envio']=='Envíos Masivos']
        Masivo.drop(['anno','trimestre','id_tipo_envio','tipo_envio','id_tipo_objeto','id_ambito'],axis=1, inplace=True)
        with st.expander('Datos masivo'):
            AgGrid(Masivo)



if select_ambito =='Internacional':
    internacional=Postales[Postales['ambito'].isin(['Internacional de salida','Internacional de entrada'])]
    select_ambinternacional = st.sidebar.selectbox('Seleccionar ámbito internacional',['Entrada','Salida'])
    
    if select_ambinternacional=='Entrada':
        Entrada=internacional[internacional['ambito']=='Internacional de entrada']
        
        col1, col2 ,col3= st.columns(3)
        with col1:
            select_objeto=st.selectbox('Seleccionar tipo de objeto',['Documentos','Paquetes'])
        with col2:    
            select_dimension = st.selectbox('Seleccione ámbito aplicación',['Nacional','Municipal'])
        with col3:
            select_variable = st.selectbox('Seleccione la variable',['Envíos','Ingresos'])
            
        if select_objeto=='Documentos':
            dfIngresos=[];dfIngresos2=[];dfIngresos3=[];
            dfEnvios=[];dfEnvios2=[];dfEnvios3=[];
            Documentos=Entrada[Entrada['tipo_objeto']=='Documentos']
            Documentos.drop(['anno','trimestre','id_tipo_envio','tipo_envio','id_tipo_objeto','tipo_objeto','id_ambito'],axis=1, inplace=True)
            PERIODOS=['2020-T3','2020-T4','2021-T1','2021-T2']
            DocumentosnacIng=Documentos.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
            DocumentosnacEnv=Documentos.groupby(['periodo','empresa','id_empresa'])['numero_total_envios'].sum().reset_index()
            
            with st.expander('Datos documentos'):
                AgGrid(Documentos)

            if select_dimension == 'Nacional':            
                select_indicador = st.sidebar.selectbox('Indicador',['Stenbacka', 'Concentración','IHH','Linda','Penetración'])
            ## Información sobre los indicadores                                
                if select_indicador == 'Stenbacka':
                    st.write("### Índice de Stenbacka")
                    st.markdown("Este índice de dominancia es una medida para identificar cuándo una empresa podría tener posición dominante en un mercado determinado. Se considera la participación de mercado de las dos empresas con mayor participación y se calcula un umbral de cuota de mercado después del cual la empresa lider posiblemente ostentaría posición de dominio. Cualquier couta de mercado superior a dicho umbral podría significar una dominancia en el mercado.")
                    #st.latex(r'''S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]''')       
                    with st.expander("Información adicional índice de Stenbacka"):
                        st.write(r""" El índice de Stenbacka está dado por la siguiente ecuación""")
                        st.latex(r"""S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]""")
                        st.write(r"""
    **Donde**
    -   $S^{2}_{1}$ y $S^{2}_{2}$ Corresponden a las participaciones de mercado de las dos empresas más grandes, respectivamente.
    -   $\gamma$ es un parámetro de competencia que puede incluir aspectos como: existencia de compradores con poder de mercado, regulación económica, presencia de derechos de propiedad, barreras a la entrada, entre otros (Lis-Guitiérrez, 2013).                
                    """,unsafe_allow_html=True)
                if select_indicador == 'Concentración':
                    st.write("### Razón de concentración")
                    st.markdown("La razón de concentración es un índice que mide las participaciones acumuladas de las empresas lideres en el mercado. Toma valores entre 0 y 1.")            
                    with st.expander("Información adicional razón de concentración"):
                        st.write("La concentración se calcula de la siguiente forma:")
                        st.latex(r''' CR_{n}=S_1+S_2+S_3+...+S_n=\sum_{i=1}^{n}S_{i}''')
                        st.write(r""" **Donde**:
    -   $S_{i}$ es la participación de mercado de la i-ésima empresa.
    -   $n$ es el número total de empresas consideradas.

    De acuerdo con Stazhkova, Kotcofana & Protasov (2017), para un $n = 3$ se pueden considerar los siguientes rangos de concentración para un mercado:

    | Concetración | Rango         |
    |--------------|---------------|
    | Baja         | $<0,45$       |
    | Moderada     | $0,45 - 0,70$ |
    | Alta         | $>0,70$       |
                    
                    
    """)
                if select_indicador == 'IHH':
                    st.write("### Índice de Herfindahl-Hirschman")
                    st.markdown("El IHH es el índice más aceptado como medida de concentración de la oferta en un mercado. Su cálculo se expresa como la suma de los cuadrados de las participaciones de las empresas que componen el mercado. El índice máximo se obtiene para un monopolio y corresponde a 10000.")            
                    with st.expander("Información adicional IHH"):
                        st.write("La fórmula del IHH está dada como")
                        st.latex(r'''IHH=\sum_{i=1}^{n}S_{i}^{2}''')
                        st.write(r"""**Donde:**
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    De acuerdo con el Departamento de Justicia y la Comisión Federal de Comercio de Estados Unidos (2010), se puede categorizar a un mercado de acuerdo a los siguientes rangos de este índice:

    | Mercado                   | Rango          |
    |---------------------------|----------------|
    | Muy competitivo           | $<100$         |
    | Desconcentrado            | $100 - 1500$   |
    | Moderadamente concentrado | $>1500 - 2500$ |
    | Altamente concentrado     | $>2500$        |                
                    """)
                if select_indicador == 'Linda':
                    st.write("### Índice de Linda")               
                    st.markdown("Este índice es utilizado para medir la desigualdad entre diferentes cuotas de mercado e identificar posibles oligopolios. El índice tomará valores cercanos a 1 en la medida que la participación en el mercado del grupo de empresas grandes es mayor que la participación del grupo de empresas pequeñas.")                    
                    with st.expander("Información adicional indicador de linda"): 
                        st.write("El indicador de Linda está dado por la siguiente ecuación:")
                        st.latex(r'''L = \frac{1}{N(N-1)} \sum_{i=1}^{N-1} (\frac{\overline{X}_{i}}{\overline{X}_{N-i}})''')
                        st.write(r"""**Donde**:
    - $\overline{X}_{i}$ es la participación de mercado media de las primeras i-ésimas empresas.
    - $\overline{X}_{N-i}$ es la partipación de mercado media de las i-ésimas empresas restantes.

    De acuerdo con Martinez (2017), se pueden considerar los siguientes rangos de concentración para un mercado:

    | Concentración   | Rango         |
    |-----------------|---------------|
    | Baja            | $<0,20$       |
    | Moderada        | $0,20 - 0,50$ |
    | Concentrada     | $>0,50 - 1$   |
    | Alta            | $>1$          |""",unsafe_allow_html=True)        
                if select_indicador == 'Penetración':
                    st.write("### Índice de penetración")
                    st.markdown("El índice de penetración es usado para...")    

            ## Cálculo de los indicadores
                if select_indicador == 'Stenbacka':
                    gamma=st.slider('Seleccionar valor gamma',0.0,2.0,0.1)
                    for elem in PERIODOS:
                        prIn=DocumentosnacIng[DocumentosnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
                        prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
                        dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))

                        prEn=DocumentosnacEnv[DocumentosnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',Participacion(prEn,'numero_total_envios'))
                        prEn.insert(4,'stenbacka',Stenbacka(prEn,'numero_total_envios',gamma))
                        dfEnvios.append(prEn.sort_values(by='participacion',ascending=False))                        
                        
                    InggroupPart=pd.concat(dfIngresos)
                    InggroupPart.participacion=InggroupPart.participacion.round(5)
                    EnvgroupPart=pd.concat(dfEnvios)
                    EnvgroupPart.participacion=EnvgroupPart.participacion.round(5)
                    if select_variable == 'Ingresos':
                        AgGrid(InggroupPart)
                        fig1=PlotlyStenbacka(InggroupPart)
                        st.plotly_chart(fig1, use_container_width=True)
                    if select_variable == 'Envíos':
                        AgGrid(EnvgroupPart)
                        fig2=PlotlyStenbacka(EnvgroupPart)
                        st.plotly_chart(fig2, use_container_width=True)                        

                if select_indicador == 'Concentración':
                    dflistEnv=[];dflistIng=[]
                    
                    for elem in PERIODOS:
                        dflistEnv.append(Concentracion(DocumentosnacEnv,'numero_total_envios',elem))
                        dflistIng.append(Concentracion(DocumentosnacIng,'ingresos',elem))
                    ConcEnv=pd.concat(dflistEnv).fillna(1.0)
                    ConcIng=pd.concat(dflistIng).fillna(1.0)
                                             
                    if select_variable == "Envíos":
                        colsconEnv=ConcEnv.columns.values.tolist()
                        value1= len(colsconEnv)-1 if len(colsconEnv)-1 >1 else 2
                        conc=st.slider('Seleccionar el número de empresas',1,value1,1,1)
                        fig4=PlotlyConcentracion(ConcEnv)
                        st.write(ConcEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconEnv[conc]]))
                        st.plotly_chart(fig4,use_container_width=True)
                    if select_variable == "Ingresos":
                        colsconIng=ConcIng.columns.values.tolist()
                        value1= len(colsconIng)-1 if len(colsconIng)-1 >1 else 2
                        conc=st.slider('Seleccione el número de empresas',1,value1,1,1)
                        fig5=PlotlyConcentracion(ConcIng)
                        st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                        st.plotly_chart(fig5,use_container_width=True)                    

                if select_indicador == 'IHH':
                    for elem in PERIODOS:
                        prEn=DocumentosnacEnv[DocumentosnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',(prEn['numero_total_envios']/prEn['numero_total_envios'].sum())*100)
                        prEn.insert(4,'IHH',IHH(prEn,'numero_total_envios'))
                        dfEnvios3.append(prEn.sort_values(by='participacion',ascending=False))
                        ##
                        prIn=DocumentosnacIng[DocumentosnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                        prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                        dfIngresos3.append(prIn.sort_values(by='participacion',ascending=False))
                        ##

                    EnvgroupPart3=pd.concat(dfEnvios3)
                    InggroupPart3=pd.concat(dfIngresos3)
                    
                    IHHEnv=EnvgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
                    IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()                
                    
                    ##Gráficas
                    
                    fig7 = PlotlyIHH(IHHEnv)   
                    fig8 = PlotlyIHH(IHHIng)  
                    
                    if select_variable == "Envíos":
                        AgGrid(EnvgroupPart3)
                        st.plotly_chart(fig7,use_container_width=True)
                    if select_variable == "Ingresos":
                        AgGrid(InggroupPart3)
                        st.plotly_chart(fig8,use_container_width=True)

                if select_indicador == 'Linda':
                    dflistEnv2=[];dflistIng2=[]                    
                    for elem in PERIODOS:
                        dflistEnv2.append(Linda(DocumentosnacEnv,'numero_total_envios',elem))
                        dflistIng2.append(Linda(DocumentosnacIng,'ingresos',elem))
                    LindEnv=pd.concat(dflistEnv2).reset_index().drop('index',axis=1).fillna(np.nan)
                    LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
         
                    if select_variable == "Envíos":
                        LindconEnv=LindEnv.columns.values.tolist()
                        lind=st.slider('Seleccionar nivel',2,len(LindconEnv),2,1)
                        fig10=PlotlyLinda(LindEnv)
                        st.write(LindEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconEnv[lind-1]]))
                        st.plotly_chart(fig10,use_container_width=True)
                    if select_variable == "Ingresos":
                        LindconIng=LindIng.columns.values.tolist()            
                        lind=st.slider('Seleccionar nivel',2,len(LindconIng),2,1)
                        fig11=PlotlyLinda(LindIng)
                        st.write(LindIng.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconIng[lind-1]]))
                        st.plotly_chart(fig11,use_container_width=True)

                if select_indicador == 'Penetración':
                    PersonasNac=Personas.groupby(['anno'])['poblacion'].sum()  
                    EnvNac=DocumentosnacEnv.groupby(['periodo'])['numero_total_envios'].sum().reset_index()
                    EnvNac.insert(0,'anno',EnvNac.periodo.str.split('-',expand=True)[0])
                    PenetracionNac=EnvNac.merge(PersonasNac, on=['anno'], how='left')
                    PenetracionNac.insert(4,'penetracion',PenetracionNac['numero_total_envios']/PenetracionNac['poblacion'])
                    PenetracionNac.penetracion=PenetracionNac.penetracion.round(3)
                    if select_variable=='Envíos':
                        fig12=PlotlyPenetracion(PenetracionNac)
                        AgGrid(PenetracionNac[['periodo','numero_total_envios','poblacion','penetracion']])
                        st.plotly_chart(fig12,use_container_width=True)
                    if select_variable=='Ingresos':
                        st.write("El indicador de penetración sólo está definido para la variable de Envíos.")   
           
        if select_objeto=='Paquetes':
            dfIngresos=[];dfIngresos2=[];dfIngresos3=[];
            dfEnvios=[];dfEnvios2=[];dfEnvios3=[];         
            Paquetes=Entrada[Entrada['tipo_objeto']=='Paquetes']
            Paquetes.drop(['anno','trimestre','id_tipo_envio','tipo_envio','id_tipo_objeto','tipo_objeto','id_ambito'],axis=1, inplace=True)
            with st.expander('Datos paquetes'):
                AgGrid(Paquetes)     
            PESO = st.select_slider('Seleccione rango de peso',Paquetes.rango_peso_envio.unique().tolist())   
            PERIODOS=['2020-T3','2020-T4','2021-T1','2021-T2']            
            PaquetesPeso=Paquetes[Paquetes['rango_peso_envio']==PESO]
            PaquetesnacIng=PaquetesPeso.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
            PaquetesnacEnv=PaquetesPeso.groupby(['periodo','empresa','id_empresa'])['numero_total_envios'].sum().reset_index()                
                
            if select_dimension == 'Nacional':       
                select_indicador = st.sidebar.selectbox('Indicador',['Stenbacka', 'Concentración','IHH','Linda','Penetración'])
                ## Información sobre los indicadores
                if select_indicador == 'Stenbacka':
                    st.write("### Índice de Stenbacka")
                    st.markdown("Este índice de dominancia es una medida para identificar cuándo una empresa podría tener posición dominante en un mercado determinado. Se considera la participación de mercado de las dos empresas con mayor participación y se calcula un umbral de cuota de mercado después del cual la empresa lider posiblemente ostentaría posición de dominio. Cualquier couta de mercado superior a dicho umbral podría significar una dominancia en el mercado.")
                    #st.latex(r'''S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]''')       
                    with st.expander("Información adicional índice de Stenbacka"):
                        st.write(r""" El índice de Stenbacka está dado por la siguiente ecuación""")
                        st.latex(r"""S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]""")
                        st.write(r"""
**Donde**
-   $S^{2}_{1}$ y $S^{2}_{2}$ Corresponden a las participaciones de mercado de las dos empresas más grandes, respectivamente.
-   $\gamma$ es un parámetro de competencia que puede incluir aspectos como: existencia de compradores con poder de mercado, regulación económica, presencia de derechos de propiedad, barreras a la entrada, entre otros (Lis-Guitiérrez, 2013).                
                """,unsafe_allow_html=True)
                if select_indicador == 'Concentración':
                    st.write("### Razón de concentración")
                    st.markdown("La razón de concentración es un índice que mide las participaciones acumuladas de las empresas lideres en el mercado. Toma valores entre 0 y 1.")            
                    with st.expander("Información adicional razón de concentración"):
                        st.write("La concentración se calcula de la siguiente forma:")
                        st.latex(r''' CR_{n}=S_1+S_2+S_3+...+S_n=\sum_{i=1}^{n}S_{i}''')
                        st.write(r""" **Donde**:
-   $S_{i}$ es la participación de mercado de la i-ésima empresa.
-   $n$ es el número total de empresas consideradas.

De acuerdo con Stazhkova, Kotcofana & Protasov (2017), para un $n = 3$ se pueden considerar los siguientes rangos de concentración para un mercado:

| Concetración | Rango         |
|--------------|---------------|
| Baja         | $<0,45$       |
| Moderada     | $0,45 - 0,70$ |
| Alta         | $>0,70$       |
                
                
""")
                if select_indicador == 'IHH':
                    st.write("### Índice de Herfindahl-Hirschman")
                    st.markdown("El IHH es el índice más aceptado como medida de concentración de la oferta en un mercado. Su cálculo se expresa como la suma de los cuadrados de las participaciones de las empresas que componen el mercado. El índice máximo se obtiene para un monopolio y corresponde a 10000.")            
                    with st.expander("Información adicional IHH"):
                        st.write("La fórmula del IHH está dada como")
                        st.latex(r'''IHH=\sum_{i=1}^{n}S_{i}^{2}''')
                        st.write(r"""**Donde:**
-   $S_{i}$ es la participación de mercado de la variable analizada.
-   $n$ es el número de empresas más grandes consideradas.

De acuerdo con el Departamento de Justicia y la Comisión Federal de Comercio de Estados Unidos (2010), se puede categorizar a un mercado de acuerdo a los siguientes rangos de este índice:

| Mercado                   | Rango          |
|---------------------------|----------------|
| Muy competitivo           | $<100$         |
| Desconcentrado            | $100 - 1500$   |
| Moderadamente concentrado | $>1500 - 2500$ |
| Altamente concentrado     | $>2500$        |                
                """)
                if select_indicador == 'Linda':
                    st.write("### Índice de Linda")               
                    st.markdown("Este índice es utilizado para medir la desigualdad entre diferentes cuotas de mercado e identificar posibles oligopolios. El índice tomará valores cercanos a 1 en la medida que la participación en el mercado del grupo de empresas grandes es mayor que la participación del grupo de empresas pequeñas.")                    
                    with st.expander("Información adicional indicador de linda"): 
                        st.write("El indicador de Linda está dado por la siguiente ecuación:")
                        st.latex(r'''L = \frac{1}{N(N-1)} \sum_{i=1}^{N-1} (\frac{\overline{X}_{i}}{\overline{X}_{N-i}})''')
                        st.write(r"""**Donde**:
- $\overline{X}_{i}$ es la participación de mercado media de las primeras i-ésimas empresas.
- $\overline{X}_{N-i}$ es la partipación de mercado media de las i-ésimas empresas restantes.

De acuerdo con Martinez (2017), se pueden considerar los siguientes rangos de concentración para un mercado:

| Concentración   | Rango         |
|-----------------|---------------|
| Baja            | $<0,20$       |
| Moderada        | $0,20 - 0,50$ |
| Concentrada     | $>0,50 - 1$   |
| Alta            | $>1$          |""",unsafe_allow_html=True)
                if select_indicador == 'Penetración':
                    st.write("### Índice de penetración")
                    st.markdown("El índice de penetración es usado para...")    

                ## Cálculo de los indicadores
                if select_indicador == 'Stenbacka':
                    gamma=st.slider('Seleccionar valor gamma',0.0,2.0,0.1)
                    for elem in PERIODOS:
                        prIn=PaquetesnacIng[PaquetesnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
                        prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
                        dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))

                        prEn=PaquetesnacEnv[PaquetesnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',Participacion(prEn,'numero_total_envios'))
                        prEn.insert(4,'stenbacka',Stenbacka(prEn,'numero_total_envios',gamma))
                        dfEnvios.append(prEn.sort_values(by='participacion',ascending=False))                        
                        
                    InggroupPart=pd.concat(dfIngresos)
                    InggroupPart.participacion=InggroupPart.participacion.round(5)
                    EnvgroupPart=pd.concat(dfEnvios)
                    EnvgroupPart.participacion=EnvgroupPart.participacion.round(5)
                    if select_variable == 'Ingresos':
                        AgGrid(InggroupPart)
                        fig1=PlotlyStenbacka(InggroupPart)
                        st.plotly_chart(fig1, use_container_width=True)
                    if select_variable == 'Envíos':
                        AgGrid(EnvgroupPart)
                        fig2=PlotlyStenbacka(EnvgroupPart)
                        st.plotly_chart(fig2, use_container_width=True)  

                if select_indicador == 'Concentración':
                    dflistEnv=[];dflistIng=[]                    
                    for elem in PERIODOS:
                        dflistEnv.append(Concentracion(PaquetesnacEnv,'numero_total_envios',elem))
                        dflistIng.append(Concentracion(PaquetesnacIng,'ingresos',elem))
                    ConcEnv=pd.concat(dflistEnv).fillna(1.0)
                    ConcIng=pd.concat(dflistIng).fillna(1.0)
                    
                    if select_variable == "Envíos":
                        colsconEnv=ConcEnv.columns.values.tolist()   
                        value1= len(colsconEnv)-1 if len(colsconEnv)-1 >1 else 2
                        conc=st.slider('Seleccionar el número de empresas',1,value1,1,1)
                        fig4=PlotlyConcentracion(ConcEnv)
                        st.write(ConcEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconEnv[conc]]))
                        st.plotly_chart(fig4,use_container_width=True)
                    if select_variable == "Ingresos":
                        colsconIng=ConcIng.columns.values.tolist() 
                        value1= len(colsconIng)-1 if len(colsconIng)-1 >1 else 2
                        conc=st.slider('Seleccione el número de empresas',1,value1,1,1)
                        fig5=PlotlyConcentracion(ConcIng)
                        st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                        st.plotly_chart(fig5,use_container_width=True)                    

                if select_indicador == 'IHH':
                    for elem in PERIODOS:
                        prEn=PaquetesnacEnv[PaquetesnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',(prEn['numero_total_envios']/prEn['numero_total_envios'].sum())*100)
                        prEn.insert(4,'IHH',IHH(prEn,'numero_total_envios'))
                        dfEnvios3.append(prEn.sort_values(by='participacion',ascending=False))
                        ##
                        prIn=PaquetesnacIng[PaquetesnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                        prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                        dfIngresos3.append(prIn.sort_values(by='participacion',ascending=False))
                        ##

                    EnvgroupPart3=pd.concat(dfEnvios3)
                    InggroupPart3=pd.concat(dfIngresos3)
                    
                    IHHEnv=EnvgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
                    IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()                
                    
                    ##Gráficas
                    
                    fig7 = PlotlyIHH(IHHEnv)   
                    fig8 = PlotlyIHH(IHHIng)  
                    
                    if select_variable == "Envíos":
                        AgGrid(EnvgroupPart3)
                        st.plotly_chart(fig7,use_container_width=True)
                    if select_variable == "Ingresos":
                        AgGrid(InggroupPart3)
                        st.plotly_chart(fig8,use_container_width=True)

                if select_indicador == 'Linda':
                    dflistEnv2=[];dflistIng2=[];nempresaIng=[]; nempresaEnv=[];                     
                    for elem in PERIODOS:
                        prEnv=PaquetesnacEnv[PaquetesnacEnv['periodo']==elem]
                        nempresaEnv.append(prEnv.empresa.nunique())
                        prIng=PaquetesnacIng[PaquetesnacIng['periodo']==elem]
                        nempresaIng.append(prIng.empresa.nunique())                    
                        dflistEnv2.append(Linda(PaquetesnacEnv,'numero_total_envios',elem))
                        dflistIng2.append(Linda(PaquetesnacIng,'ingresos',elem))
                    NemphisEnv=max(nempresaIng)
                    NemphisIng=max(nempresaEnv)    
                    LindEnv=pd.concat(dflistEnv2).reset_index().drop('index',axis=1).fillna(np.nan)
                    LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
         
                    if select_variable == "Envíos":
                        LindconEnv=LindEnv.columns.values.tolist()
                        if NemphisEnv==1:
                            st.write("El índice de linda no está definido pues sólo hay una empresa")  
                        else:            
                            lind=st.slider('Seleccionar nivel',2,len(LindconEnv),2,1)
                            fig10=PlotlyLinda(LindEnv)
                            st.write(LindEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconEnv[lind-1]]))
                            st.plotly_chart(fig10,use_container_width=True)
                    if select_variable == "Ingresos":                    
                        LindconIng=LindIng.columns.values.tolist()        
                        if NemphisIng==1:
                            st.write("El índice de linda no está definido pues sólo hay una empresa")  
                        else:            
                            lind=st.slider('Seleccionar nivel',2,len(LindconIng),2,1)
                            fig11=PlotlyLinda(LindIng)
                            st.write(LindIng.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconIng[lind-1]]))
                            st.plotly_chart(fig11,use_container_width=True)

                if select_indicador == 'Penetración':
                    PersonasNac=Personas.groupby(['anno'])['poblacion'].sum()  
                    EnvNac=PaquetesnacEnv.groupby(['periodo'])['numero_total_envios'].sum().reset_index()
                    EnvNac.insert(0,'anno',EnvNac.periodo.str.split('-',expand=True)[0])
                    PenetracionNac=EnvNac.merge(PersonasNac, on=['anno'], how='left')
                    PenetracionNac.insert(4,'penetracion',PenetracionNac['numero_total_envios']/PenetracionNac['poblacion'])
                    PenetracionNac.penetracion=PenetracionNac.penetracion.round(3)
                    
                    if select_variable=='Envíos':
                        fig12=PlotlyPenetracion(PenetracionNac)
                        AgGrid(PenetracionNac[['periodo','numero_total_envios','poblacion','penetracion']])
                        st.plotly_chart(fig12,use_container_width=True)
                    if select_variable=='Ingresos':
                        st.write("El indicador de penetración sólo está definido para la variable de Envíos.")   
               
    if select_ambinternacional=='Salida':
        Salida=internacional[internacional['ambito']=='Internacional de salida']

        col1, col2 ,col3= st.columns(3)
        with col1:
            select_objeto=st.selectbox('Seleccionar tipo de objeto',['Documentos','Paquetes'])
        with col2:    
            select_dimension = st.selectbox('Seleccione ámbito aplicación',['Nacional','Municipal'])
        with col3:
            select_variable = st.selectbox('Seleccione la variable',['Envíos','Ingresos'])

        if select_objeto=='Documentos':
            dfIngresos=[];dfIngresos2=[];dfIngresos3=[];
            dfEnvios=[];dfEnvios2=[];dfEnvios3=[];        
            Documentos=Salida[Salida['tipo_objeto']=='Documentos']
            Documentos.drop(['anno','trimestre','id_tipo_envio','tipo_envio','id_tipo_objeto','tipo_objeto','id_ambito'],axis=1, inplace=True)
            PERIODOS=['2020-T3','2020-T4','2021-T1','2021-T2']
            DocumentosnacIng=Documentos.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
            DocumentosnacEnv=Documentos.groupby(['periodo','empresa','id_empresa'])['numero_total_envios'].sum().reset_index()
            with st.expander('Datos documentos'):
                AgGrid(Documentos)
            if select_dimension == 'Nacional':       
                select_indicador = st.sidebar.selectbox('Indicador',['Stenbacka', 'Concentración','IHH','Linda','Penetración'])
                ## Información sobre los indicadores

                if select_indicador == 'Stenbacka':
                    st.write("### Índice de Stenbacka")
                    st.markdown("Este índice de dominancia es una medida para identificar cuándo una empresa podría tener posición dominante en un mercado determinado. Se considera la participación de mercado de las dos empresas con mayor participación y se calcula un umbral de cuota de mercado después del cual la empresa lider posiblemente ostentaría posición de dominio. Cualquier couta de mercado superior a dicho umbral podría significar una dominancia en el mercado.")
                    #st.latex(r'''S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]''')       
                    with st.expander("Información adicional índice de Stenbacka"):
                        st.write(r""" El índice de Stenbacka está dado por la siguiente ecuación""")
                        st.latex(r"""S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]""")
                        st.write(r"""
**Donde**
-   $S^{2}_{1}$ y $S^{2}_{2}$ Corresponden a las participaciones de mercado de las dos empresas más grandes, respectivamente.
-   $\gamma$ es un parámetro de competencia que puede incluir aspectos como: existencia de compradores con poder de mercado, regulación económica, presencia de derechos de propiedad, barreras a la entrada, entre otros (Lis-Guitiérrez, 2013).                
                """,unsafe_allow_html=True)
                if select_indicador == 'Concentración':
                    st.write("### Razón de concentración")
                    st.markdown("La razón de concentración es un índice que mide las participaciones acumuladas de las empresas lideres en el mercado. Toma valores entre 0 y 1.")            
                    with st.expander("Información adicional razón de concentración"):
                        st.write("La concentración se calcula de la siguiente forma:")
                        st.latex(r''' CR_{n}=S_1+S_2+S_3+...+S_n=\sum_{i=1}^{n}S_{i}''')
                        st.write(r""" **Donde**:
-   $S_{i}$ es la participación de mercado de la i-ésima empresa.
-   $n$ es el número total de empresas consideradas.

De acuerdo con Stazhkova, Kotcofana & Protasov (2017), para un $n = 3$ se pueden considerar los siguientes rangos de concentración para un mercado:

| Concetración | Rango         |
|--------------|---------------|
| Baja         | $<0,45$       |
| Moderada     | $0,45 - 0,70$ |
| Alta         | $>0,70$       |
                
                
""")
                if select_indicador == 'IHH':
                    st.write("### Índice de Herfindahl-Hirschman")
                    st.markdown("El IHH es el índice más aceptado como medida de concentración de la oferta en un mercado. Su cálculo se expresa como la suma de los cuadrados de las participaciones de las empresas que componen el mercado. El índice máximo se obtiene para un monopolio y corresponde a 10000.")            
                    with st.expander("Información adicional IHH"):
                        st.write("La fórmula del IHH está dada como")
                        st.latex(r'''IHH=\sum_{i=1}^{n}S_{i}^{2}''')
                        st.write(r"""**Donde:**
-   $S_{i}$ es la participación de mercado de la variable analizada.
-   $n$ es el número de empresas más grandes consideradas.

De acuerdo con el Departamento de Justicia y la Comisión Federal de Comercio de Estados Unidos (2010), se puede categorizar a un mercado de acuerdo a los siguientes rangos de este índice:

| Mercado                   | Rango          |
|---------------------------|----------------|
| Muy competitivo           | $<100$         |
| Desconcentrado            | $100 - 1500$   |
| Moderadamente concentrado | $>1500 - 2500$ |
| Altamente concentrado     | $>2500$        |                
                """)
                if select_indicador == 'Linda':                
                    st.write("### Índice de Linda")               
                    st.markdown("Este índice es utilizado para medir la desigualdad entre diferentes cuotas de mercado e identificar posibles oligopolios. El índice tomará valores cercanos a 1 en la medida que la participación en el mercado del grupo de empresas grandes es mayor que la participación del grupo de empresas pequeñas.")                    
                    with st.expander("Información adicional indicador de linda"): 
                        st.write("El indicador de Linda está dado por la siguiente ecuación:")
                        st.latex(r'''L = \frac{1}{N(N-1)} \sum_{i=1}^{N-1} (\frac{\overline{X}_{i}}{\overline{X}_{N-i}})''')
                        st.write(r"""**Donde**:
- $\overline{X}_{i}$ es la participación de mercado media de las primeras i-ésimas empresas.
- $\overline{X}_{N-i}$ es la partipación de mercado media de las i-ésimas empresas restantes.

De acuerdo con Martinez (2017), se pueden considerar los siguientes rangos de concentración para un mercado:

| Concentración   | Rango         |
|-----------------|---------------|
| Baja            | $<0,20$       |
| Moderada        | $0,20 - 0,50$ |
| Concentrada     | $>0,50 - 1$   |
| Alta            | $>1$          |""",unsafe_allow_html=True)
                if select_indicador == 'Penetración':
                    st.write("### Índice de penetración")
                    st.markdown("El índice de penetración es usado para...")    

            ## Cálculo de los indicadores
                if select_indicador == 'Stenbacka':
                    gamma=st.slider('Seleccionar valor gamma',0.0,2.0,0.1)
                    for elem in PERIODOS:
                        prIn=DocumentosnacIng[DocumentosnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
                        prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
                        dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))

                        prEn=DocumentosnacEnv[DocumentosnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',Participacion(prEn,'numero_total_envios'))
                        prEn.insert(4,'stenbacka',Stenbacka(prEn,'numero_total_envios',gamma))
                        dfEnvios.append(prEn.sort_values(by='participacion',ascending=False))                        
                        
                    InggroupPart=pd.concat(dfIngresos)
                    InggroupPart.participacion=InggroupPart.participacion.round(5)
                    EnvgroupPart=pd.concat(dfEnvios)
                    EnvgroupPart.participacion=EnvgroupPart.participacion.round(5)
                    if select_variable == 'Ingresos':
                        AgGrid(InggroupPart)
                        fig1=PlotlyStenbacka(InggroupPart)
                        st.plotly_chart(fig1, use_container_width=True)
                    if select_variable == 'Envíos':
                        AgGrid(EnvgroupPart)
                        fig2=PlotlyStenbacka(EnvgroupPart)
                        st.plotly_chart(fig2, use_container_width=True)                        

                if select_indicador == 'Concentración':
                    dflistEnv=[];dflistIng=[]
                    
                    for elem in PERIODOS:
                        dflistEnv.append(Concentracion(DocumentosnacEnv,'numero_total_envios',elem))
                        dflistIng.append(Concentracion(DocumentosnacIng,'ingresos',elem))
                    ConcEnv=pd.concat(dflistEnv).fillna(1.0)
                    ConcIng=pd.concat(dflistIng).fillna(1.0)
                                             
                    if select_variable == "Envíos":
                        colsconEnv=ConcEnv.columns.values.tolist()
                        conc=st.slider('Seleccionar el número de empresas',1,len(colsconEnv)-1,1,1)
                        fig4=PlotlyConcentracion(ConcEnv)
                        st.write(ConcEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconEnv[conc]]))
                        st.plotly_chart(fig4,use_container_width=True)
                    if select_variable == "Ingresos":
                        colsconIng=ConcIng.columns.values.tolist()
                        conc=st.slider('Seleccione el número de empresas',1,len(colsconIng)-1,1,1)
                        fig5=PlotlyConcentracion(ConcIng)
                        st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                        st.plotly_chart(fig5,use_container_width=True)                    

                if select_indicador == 'IHH':
                    for elem in PERIODOS:
                        prEn=DocumentosnacEnv[DocumentosnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',(prEn['numero_total_envios']/prEn['numero_total_envios'].sum())*100)
                        prEn.insert(4,'IHH',IHH(prEn,'numero_total_envios'))
                        dfEnvios3.append(prEn.sort_values(by='participacion',ascending=False))
                        ##
                        prIn=DocumentosnacIng[DocumentosnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                        prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                        dfIngresos3.append(prIn.sort_values(by='participacion',ascending=False))
                        ##

                    EnvgroupPart3=pd.concat(dfEnvios3)
                    InggroupPart3=pd.concat(dfIngresos3)
                    
                    IHHEnv=EnvgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
                    IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()                
                    
                    ##Gráficas
                    
                    fig7 = PlotlyIHH(IHHEnv)   
                    fig8 = PlotlyIHH(IHHIng)  
                    
                    if select_variable == "Envíos":
                        AgGrid(EnvgroupPart3)
                        st.plotly_chart(fig7,use_container_width=True)
                    if select_variable == "Ingresos":
                        AgGrid(InggroupPart3)
                        st.plotly_chart(fig8,use_container_width=True)

                if select_indicador == 'Linda':
                    dflistEnv2=[];dflistIng2=[]                    
                    for elem in PERIODOS:
                        dflistEnv2.append(Linda(DocumentosnacEnv,'numero_total_envios',elem))
                        dflistIng2.append(Linda(DocumentosnacIng,'ingresos',elem))
                    LindEnv=pd.concat(dflistEnv2).reset_index().drop('index',axis=1).fillna(np.nan)
                    LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
         
                    if select_variable == "Envíos":
                        LindconEnv=LindEnv.columns.values.tolist()
                        lind=st.slider('Seleccionar nivel',2,len(LindconEnv),2,1)
                        fig10=PlotlyLinda(LindEnv)
                        st.write(LindEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconEnv[lind-1]]))
                        st.plotly_chart(fig10,use_container_width=True)
                    if select_variable == "Ingresos":
                        LindconIng=LindIng.columns.values.tolist()            
                        lind=st.slider('Seleccionar nivel',2,len(LindconIng),2,1)
                        fig11=PlotlyLinda(LindIng)
                        st.write(LindIng.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconIng[lind-1]]))
                        st.plotly_chart(fig11,use_container_width=True)

                if select_indicador == 'Penetración':
                    PersonasNac=Personas.groupby(['anno'])['poblacion'].sum()  
                    EnvNac=DocumentosnacEnv.groupby(['periodo'])['numero_total_envios'].sum().reset_index()
                    EnvNac.insert(0,'anno',EnvNac.periodo.str.split('-',expand=True)[0])
                    PenetracionNac=EnvNac.merge(PersonasNac, on=['anno'], how='left')
                    PenetracionNac.insert(4,'penetracion',PenetracionNac['numero_total_envios']/PenetracionNac['poblacion'])
                    PenetracionNac.penetracion=PenetracionNac.penetracion.round(3)
                    if select_variable=='Envíos':
                        fig12=PlotlyPenetracion(PenetracionNac)
                        AgGrid(PenetracionNac[['periodo','numero_total_envios','poblacion','penetracion']])
                        st.plotly_chart(fig12,use_container_width=True)
                    if select_variable=='Ingresos':
                        st.write("El indicador de penetración sólo está definido para la variable de Envíos.")   

                
        if select_objeto=='Paquetes':
            dfIngresos=[];dfIngresos2=[];dfIngresos3=[];
            dfEnvios=[];dfEnvios2=[];dfEnvios3=[];           
            Paquetes=Salida[Salida['tipo_objeto']=='Paquetes']
            Paquetes.drop(['anno','trimestre','id_tipo_envio','tipo_envio','id_tipo_objeto','tipo_objeto','id_ambito'],axis=1, inplace=True)
            with st.expander('Datos paquetes'):
                AgGrid(Paquetes)        
            PESO = st.select_slider('Seleccione rango de peso',Paquetes.rango_peso_envio.unique().tolist())   
            PERIODOS=['2020-T3','2020-T4','2021-T1','2021-T2']            
            PaquetesPeso=Paquetes[Paquetes['rango_peso_envio']==PESO]
            PaquetesnacIng=PaquetesPeso.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
            PaquetesnacEnv=PaquetesPeso.groupby(['periodo','empresa','id_empresa'])['numero_total_envios'].sum().reset_index()                
                
            if select_dimension == 'Nacional':       
                select_indicador = st.sidebar.selectbox('Indicador',['Stenbacka', 'Concentración','IHH','Linda','Penetración'])
                ## Información sobre los indicadores
                if select_indicador == 'Stenbacka':
                    st.write("### Índice de Stenbacka")
                    st.markdown("Este índice de dominancia es una medida para identificar cuándo una empresa podría tener posición dominante en un mercado determinado. Se considera la participación de mercado de las dos empresas con mayor participación y se calcula un umbral de cuota de mercado después del cual la empresa lider posiblemente ostentaría posición de dominio. Cualquier couta de mercado superior a dicho umbral podría significar una dominancia en el mercado.")
                    #st.latex(r'''S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]''')       
                    with st.expander("Información adicional índice de Stenbacka"):
                        st.write(r""" El índice de Stenbacka está dado por la siguiente ecuación""")
                        st.latex(r"""S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]""")
                        st.write(r"""
**Donde**
-   $S^{2}_{1}$ y $S^{2}_{2}$ Corresponden a las participaciones de mercado de las dos empresas más grandes, respectivamente.
-   $\gamma$ es un parámetro de competencia que puede incluir aspectos como: existencia de compradores con poder de mercado, regulación económica, presencia de derechos de propiedad, barreras a la entrada, entre otros (Lis-Guitiérrez, 2013).                
                """,unsafe_allow_html=True)
                if select_indicador == 'Concentración':
                    st.write("### Razón de concentración")
                    st.markdown("La razón de concentración es un índice que mide las participaciones acumuladas de las empresas lideres en el mercado. Toma valores entre 0 y 1.")            
                    with st.expander("Información adicional razón de concentración"):
                        st.write("La concentración se calcula de la siguiente forma:")
                        st.latex(r''' CR_{n}=S_1+S_2+S_3+...+S_n=\sum_{i=1}^{n}S_{i}''')
                        st.write(r""" **Donde**:
-   $S_{i}$ es la participación de mercado de la i-ésima empresa.
-   $n$ es el número total de empresas consideradas.

De acuerdo con Stazhkova, Kotcofana & Protasov (2017), para un $n = 3$ se pueden considerar los siguientes rangos de concentración para un mercado:

| Concetración | Rango         |
|--------------|---------------|
| Baja         | $<0,45$       |
| Moderada     | $0,45 - 0,70$ |
| Alta         | $>0,70$       |
                
                
""")
                if select_indicador == 'IHH':
                    st.write("### Índice de Herfindahl-Hirschman")
                    st.markdown("El IHH es el índice más aceptado como medida de concentración de la oferta en un mercado. Su cálculo se expresa como la suma de los cuadrados de las participaciones de las empresas que componen el mercado. El índice máximo se obtiene para un monopolio y corresponde a 10000.")            
                    with st.expander("Información adicional IHH"):
                        st.write("La fórmula del IHH está dada como")
                        st.latex(r'''IHH=\sum_{i=1}^{n}S_{i}^{2}''')
                        st.write(r"""**Donde:**
-   $S_{i}$ es la participación de mercado de la variable analizada.
-   $n$ es el número de empresas más grandes consideradas.

De acuerdo con el Departamento de Justicia y la Comisión Federal de Comercio de Estados Unidos (2010), se puede categorizar a un mercado de acuerdo a los siguientes rangos de este índice:

| Mercado                   | Rango          |
|---------------------------|----------------|
| Muy competitivo           | $<100$         |
| Desconcentrado            | $100 - 1500$   |
| Moderadamente concentrado | $>1500 - 2500$ |
| Altamente concentrado     | $>2500$        |                
                """)
                if select_indicador == 'Linda':               
                    st.write("### Índice de Linda")               
                    st.markdown("Este índice es utilizado para medir la desigualdad entre diferentes cuotas de mercado e identificar posibles oligopolios. El índice tomará valores cercanos a 1 en la medida que la participación en el mercado del grupo de empresas grandes es mayor que la participación del grupo de empresas pequeñas.")                    
                    with st.expander("Información adicional indicador de linda"): 
                        st.write("El indicador de Linda está dado por la siguiente ecuación:")
                        st.latex(r'''L = \frac{1}{N(N-1)} \sum_{i=1}^{N-1} (\frac{\overline{X}_{i}}{\overline{X}_{N-i}})''')
                        st.write(r"""**Donde**:
- $\overline{X}_{i}$ es la participación de mercado media de las primeras i-ésimas empresas.
- $\overline{X}_{N-i}$ es la partipación de mercado media de las i-ésimas empresas restantes.

De acuerdo con Martinez (2017), se pueden considerar los siguientes rangos de concentración para un mercado:

| Concentración   | Rango         |
|-----------------|---------------|
| Baja            | $<0,20$       |
| Moderada        | $0,20 - 0,50$ |
| Concentrada     | $>0,50 - 1$   |
| Alta            | $>1$          |""",unsafe_allow_html=True)
                if select_indicador == 'Penetración':
                    st.write("### Índice de penetración")
                    st.markdown("El índice de penetración es usado para...")

                ## Cálculo de los indicadores

                if select_indicador == 'Stenbacka':
                    gamma=st.slider('Seleccionar valor gamma',0.0,2.0,0.1)
                    for elem in PERIODOS:
                        prIn=PaquetesnacIng[PaquetesnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
                        prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
                        dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))

                        prEn=PaquetesnacEnv[PaquetesnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',Participacion(prEn,'numero_total_envios'))
                        prEn.insert(4,'stenbacka',Stenbacka(prEn,'numero_total_envios',gamma))
                        dfEnvios.append(prEn.sort_values(by='participacion',ascending=False))                        
                        
                    InggroupPart=pd.concat(dfIngresos)
                    InggroupPart.participacion=InggroupPart.participacion.round(5)
                    EnvgroupPart=pd.concat(dfEnvios)
                    EnvgroupPart.participacion=EnvgroupPart.participacion.round(5)
                    if select_variable == 'Ingresos':
                        AgGrid(InggroupPart)
                        fig1=PlotlyStenbacka(InggroupPart)
                        st.plotly_chart(fig1, use_container_width=True)
                    if select_variable == 'Envíos':
                        AgGrid(EnvgroupPart)
                        fig2=PlotlyStenbacka(EnvgroupPart)
                        st.plotly_chart(fig2, use_container_width=True)  

                if select_indicador == 'Concentración':
                    dflistEnv=[];dflistIng=[]                    
                    for elem in PERIODOS:
                        dflistEnv.append(Concentracion(PaquetesnacEnv,'numero_total_envios',elem))
                        dflistIng.append(Concentracion(PaquetesnacIng,'ingresos',elem))
                    ConcEnv=pd.concat(dflistEnv).fillna(1.0)
                    ConcIng=pd.concat(dflistIng).fillna(1.0)
                    
                    if select_variable == "Envíos":
                        colsconEnv=ConcEnv.columns.values.tolist()   
                        value1= len(colsconEnv)-1 if len(colsconEnv)-1 >1 else 2
                        conc=st.slider('Seleccionar el número de empresas',1,value1,1,1)
                        fig4=PlotlyConcentracion(ConcEnv)
                        st.write(ConcEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconEnv[conc]]))
                        st.plotly_chart(fig4,use_container_width=True)
                    if select_variable == "Ingresos":
                        colsconIng=ConcIng.columns.values.tolist() 
                        value1= len(colsconIng)-1 if len(colsconIng)-1 >1 else 2
                        conc=st.slider('Seleccione el número de empresas',1,value1,1,1)
                        fig5=PlotlyConcentracion(ConcIng)
                        st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                        st.plotly_chart(fig5,use_container_width=True)                    

                if select_indicador == 'IHH':
                    for elem in PERIODOS:
                        prEn=PaquetesnacEnv[PaquetesnacEnv['periodo']==elem]
                        prEn.insert(3,'participacion',(prEn['numero_total_envios']/prEn['numero_total_envios'].sum())*100)
                        prEn.insert(4,'IHH',IHH(prEn,'numero_total_envios'))
                        dfEnvios3.append(prEn.sort_values(by='participacion',ascending=False))
                        ##
                        prIn=PaquetesnacIng[PaquetesnacIng['periodo']==elem]
                        prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                        prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                        dfIngresos3.append(prIn.sort_values(by='participacion',ascending=False))
                        ##

                    EnvgroupPart3=pd.concat(dfEnvios3)
                    InggroupPart3=pd.concat(dfIngresos3)
                    
                    IHHEnv=EnvgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
                    IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()                
                    
                    ##Gráficas
                    
                    fig7 = PlotlyIHH(IHHEnv)   
                    fig8 = PlotlyIHH(IHHIng)  
                    
                    if select_variable == "Envíos":
                        AgGrid(EnvgroupPart3)
                        st.plotly_chart(fig7,use_container_width=True)
                    if select_variable == "Ingresos":
                        AgGrid(InggroupPart3)
                        st.plotly_chart(fig8,use_container_width=True)

                if select_indicador == 'Linda':
                    dflistEnv2=[];dflistIng2=[];nempresaIng=[]; nempresaEnv=[];                     
                    for elem in PERIODOS:
                        prEnv=PaquetesnacEnv[PaquetesnacEnv['periodo']==elem]
                        nempresaEnv.append(prEnv.empresa.nunique())
                        prIng=PaquetesnacIng[PaquetesnacIng['periodo']==elem]
                        nempresaIng.append(prIng.empresa.nunique())                    
                        dflistEnv2.append(Linda(PaquetesnacEnv,'numero_total_envios',elem))
                        dflistIng2.append(Linda(PaquetesnacIng,'ingresos',elem))
                    NemphisEnv=max(nempresaIng)
                    NemphisIng=max(nempresaEnv)    
                    LindEnv=pd.concat(dflistEnv2).reset_index().drop('index',axis=1).fillna(np.nan)
                    LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
         
                    if select_variable == "Envíos":
                        LindconEnv=LindEnv.columns.values.tolist()
                        if NemphisEnv==1:
                            st.write("El índice de linda no está definido pues sólo hay una empresa")  
                        else:            
                            lind=st.slider('Seleccionar nivel',2,len(LindconEnv),2,1)
                            fig10=PlotlyLinda(LindEnv)
                            st.write(LindEnv.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconEnv[lind-1]]))
                            st.plotly_chart(fig10,use_container_width=True)
                    if select_variable == "Ingresos":                    
                        LindconIng=LindIng.columns.values.tolist()        
                        if NemphisIng==1:
                            st.write("El índice de linda no está definido pues sólo hay una empresa")  
                        else:            
                            lind=st.slider('Seleccionar nivel',2,len(LindconIng),2,1)
                            fig11=PlotlyLinda(LindIng)
                            st.write(LindIng.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconIng[lind-1]]))
                            st.plotly_chart(fig11,use_container_width=True)

                if select_indicador == 'Penetración':
                    PersonasNac=Personas.groupby(['anno'])['poblacion'].sum()  
                    EnvNac=PaquetesnacEnv.groupby(['periodo'])['numero_total_envios'].sum().reset_index()
                    EnvNac.insert(0,'anno',EnvNac.periodo.str.split('-',expand=True)[0])
                    PenetracionNac=EnvNac.merge(PersonasNac, on=['anno'], how='left')
                    PenetracionNac.insert(4,'penetracion',PenetracionNac['numero_total_envios']/PenetracionNac['poblacion'])
                    PenetracionNac.penetracion=PenetracionNac.penetracion.round(3)
                    
                    if select_variable=='Envíos':
                        fig12=PlotlyPenetracion(PenetracionNac)
                        AgGrid(PenetracionNac[['periodo','numero_total_envios','poblacion','penetracion']])
                        st.plotly_chart(fig12,use_container_width=True)
                    if select_variable=='Ingresos':
                        st.write("El indicador de penetración sólo está definido para la variable de Envíos.")   

  
            