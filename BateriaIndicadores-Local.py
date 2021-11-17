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
    empresasdf=df['empresa'].unique().tolist()
    fig = make_subplots(rows=1, cols=1)
    dfStenbacka=df.groupby(['periodo'])['stenbacka'].mean().reset_index()
    for elem in empresasdf:
        fig.add_trace(go.Scatter(x=df[df['empresa']==elem]['periodo'],
        y=df[df['empresa']==elem]['participacion'],
        mode='lines+markers',line = dict(width=0.8),name='',hovertemplate =
        '<br><b>Empresa</b>:<br>'+elem+
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>Participación</b>: %{y:.4f}<br>')) 
    fig.add_trace(go.Scatter(x=dfStenbacka['periodo'],y=dfStenbacka['stenbacka'],name='',marker_color='rgba(128, 128, 128, 0.5)',fill='tozeroy',fillcolor='rgba(192, 192, 192, 0.15)',
        hovertemplate =
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>Stenbacka</b>: %{y:.4f}<br>'))    
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text="PERIODO",row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="PARTICIPACIÓN", row=1, col=1)
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
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text="PERIODO",row=1, col=1)
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

def PlotlyIHH(df):    
    fig = make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=df['IHH'],
                         hovertemplate =
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>IHH</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text="PERIODO",row=1, col=1)
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
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text="PERIODO",row=1, col=1)
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
#    fig = go.Figure(data=[go.Pie(labels=df.municipio.values.tolist(),
#                             values=df.WJ.values.tolist())])
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=20, uniformtext_mode='hide',height=300, width=300)
    fig.update_traces(hoverinfo='label+percent', textinfo='value',
                  marker=dict(line=dict(color='#000000', width=1)))
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))    
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.96))
    return fig

def PlotlyLinda(df):    
    fig = make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=flatten(df.iloc[:, [lind-1]].values),hovertemplate =
    '<br><b>Periodo</b>: %{x}<br>'+                         
    '<br><b>Linda</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text="PERIODO",row=1, col=1)
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
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text="PERIODO",row=1, col=1)
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
# Set page title and favicon.

st.set_page_config(
    page_title="Batería de indicadores", page_icon=LogoComision,layout="wide",initial_sidebar_state="expanded")
     
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
st.markdown(r""" **<center><ins>Guía de uso de la batería de indicadores para el análisis de competencia</ins></center>**
- Use el menú de la barra de la izquierda para seleccionar el mercado sobre el cuál le gustaría realizar el cálculo de los indicadores.
- Elija el ámbito del mercado: Departamental, Municipal, Nacional.
- Escoja el indicador a calcular.
- Dependiendo del ámbito y el indicador, interactúe con los parámetros establecidos, tal como periodo, municipio, número de empresas, etc.
""",unsafe_allow_html=True)  
st.sidebar.markdown("""<b>Seleccione el indicador a calcular</b>""", unsafe_allow_html=True)

select_mercado = st.sidebar.selectbox('Mercado',
                                    ['Telefonía local', 'Internet fijo','Televisión por suscripción'])
                              
#API 
consulta_anno = '2017,2018,2019,2020,2021,2022,2023,2024,2025'
##TELEFONIA LOCAL
   ###TRAFICO
@st.cache(allow_output_mutation=True)
def ReadAPITrafTL():
    resourceid_tl_traf = 'bb2b4afe-f098-4c5d-819a-cba76337c3a9'
    consulta_tl_traf='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid_tl_traf + ''\
                        '&filters[anno]=' + consulta_anno + ''\
                        '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa&fields[]=empresa&fields[]=id_departamento&fields[]=departamento&fields[]=id_municipio&fields[]=municipio'\
                        '&group_by=anno,trimestre,id_empresa,empresa,id_departamento,departamento,id_municipio,municipio'\
                        '&sum=trafico' 
    response_tl_traf = urlopen(consulta_tl_traf + '&limit=10000000') # Se obtiene solo un registro para obtener el total de registros en la respuesta
    json_tl_traf = json.loads(response_tl_traf.read())
    TL_TRAF = pd.DataFrame(json_tl_traf['result']['records'])
    TL_TRAF.sum_trafico = TL_TRAF.sum_trafico.astype('int64')
    TL_TRAF = TL_TRAF.rename(columns={'sum_trafico':'trafico'})
    return TL_TRAF
   ###LINEAS
@st.cache(allow_output_mutation=True)
def ReadAPILinTL():
    resourceid_tl_lineas = '967fbbd1-1c10-42b8-a6af-88b2376d43e7'
    consulta_tl_lineas = 'https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid_tl_lineas + ''\
                        '&filters[anno]=' + consulta_anno + ''\
                        '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa&fields[]=empresa&fields[]=id_departamento&fields[]=departamento&fields[]=id_municipio&fields[]=municipio'\
                        '&group_by=anno,trimestre,id_empresa,empresa,id_departamento,departamento,id_municipio,municipio'\
                        '&sum=lineas' 
    response_tl_lineas = urlopen(consulta_tl_lineas + '&limit=10000000') # Se obtiene solo un registro para obtener el total de registros en la respuesta
    json_tl_lineas = json.loads(response_tl_lineas.read())
    TL_LINEAS = pd.DataFrame(json_tl_lineas['result']['records'])
    TL_LINEAS.sum_lineas = TL_LINEAS.sum_lineas.astype('int64')
    TL_LINEAS = TL_LINEAS.rename(columns={'sum_lineas':'lineas'})
    return TL_LINEAS    
    ###INGRESOS
@st.cache(allow_output_mutation=True)
def ReadAPIIngTL():
    resourceid_tl_ing = 'f923f3bc-0628-44cc-beed-ca98b8bc3679'
    consulta_tl_ing = 'https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid_tl_ing + ''\
                        '&filters[anno]=' + consulta_anno + ''\
                        '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
                        '&group_by=anno,trimestre,id_empresa,empresa'\
                        '&sum=ingresos' 
    response_tl_ing = urlopen(consulta_tl_ing + '&limit=10000000') # Se obtiene solo un registro para obtener el total de registros en la respuesta
    json_tl_ing = json.loads(response_tl_ing.read())
    TL_ING = pd.DataFrame(json_tl_ing['result']['records'])
    TL_ING.sum_ingresos = TL_ING.sum_ingresos.astype('int64')
    TL_ING = TL_ING.rename(columns={'sum_ingresos':'ingresos'})
    return TL_ING    



if select_mercado == 'Telefonía local':   
    st.title('Telefonía local') 
    Trafico=ReadAPITrafTL()
    Ingresos=ReadAPIIngTL()
    Lineas=ReadAPILinTL()
    Trafico['periodo']=Trafico['anno']+'-T'+Trafico['trimestre']
    Ingresos['periodo']=Ingresos['anno']+'-T'+Ingresos['trimestre']
    Lineas['periodo']=Lineas['anno']+'-T'+Lineas['trimestre']
    Trafnac=Trafico.groupby(['periodo','empresa'])['trafico'].sum().reset_index()
    Ingnac=Ingresos.groupby(['periodo','empresa'])['ingresos'].sum().reset_index()
    Linnac=Lineas.groupby(['periodo','empresa'])['lineas'].sum().reset_index()
    PERIODOS=Trafnac['periodo'].unique().tolist()
    
    Trafdpto=Trafico.groupby(['periodo','id_departamento','departamento','empresa'])['trafico'].sum().reset_index()
    Trafdpto=Trafdpto[Trafdpto['trafico']>0]
    Lindpto=Lineas.groupby(['periodo','id_departamento','departamento','empresa'])['lineas'].sum().reset_index()
    Lindpto=Lindpto[Lindpto['lineas']>0]

    
    Trafmuni=Trafico.groupby(['periodo','id_municipio','municipio','departamento','empresa'])['trafico'].sum().reset_index()
    Trafmuni=Trafmuni[Trafmuni['trafico']>0]
    Trafmuni.insert(1,'codigo',Trafmuni['municipio']+' - '+Trafmuni['id_municipio'])
    Trafmuni=Trafmuni.drop(['id_municipio','municipio'],axis=1)
    Linmuni=Lineas.groupby(['periodo','id_municipio','municipio','departamento','empresa'])['lineas'].sum().reset_index()
    Linmuni=Linmuni[Linmuni['lineas']>0]
    Linmuni.insert(1,'codigo',Linmuni['municipio']+' - '+Linmuni['id_municipio'])
    Linmuni=Linmuni.drop(['id_municipio','municipio'],axis=1)
    dfTrafico=[];dfIngresos=[];dfLineas=[]
    dfTrafico2=[];dfIngresos2=[];dfLineas2=[]
    dfTrafico3=[];dfIngresos3=[];dfLineas3=[]
    
    select_dimension=st.sidebar.selectbox('Ámbito',['Departamental','Municipal','Nacional'])
    
    if select_dimension == 'Nacional':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda'])
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
    
        st.write('#### Agregación nacional') 
        select_variable = st.selectbox('Variable',['Tráfico', 'Ingresos','Líneas']) 
        
    ## Cálculo de los indicadores    
        if select_indicador == 'Stenbacka':
            gamma=st.slider('Seleccionar valor gamma',0.0,1.0,0.1)
            for elem in PERIODOS:
                prTr=Trafnac[Trafnac['periodo']==elem]
                prTr.insert(3,'participacion',Participacion(prTr,'trafico'))
                prTr.insert(4,'stenbacka',Stenbacka(prTr,'trafico',gamma))
                dfTrafico.append(prTr.sort_values(by='participacion',ascending=False))
        
                prIn=Ingnac[Ingnac['periodo']==elem]
                prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
                prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
                dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))
        
                prLi=Linnac[Linnac['periodo']==elem]
                prLi.insert(3,'participacion',Participacion(prLi,'lineas'))
                prLi.insert(4,'stenbacka',Stenbacka(prLi,'lineas',gamma))
                dfLineas.append(prLi.sort_values(by='participacion',ascending=False)) 
            TrafgroupPart=pd.concat(dfTrafico)
            InggroupPart=pd.concat(dfIngresos)
            LingroupPart=pd.concat(dfLineas)

            #Gráficas
            fig1=PlotlyStenbacka(TrafgroupPart)
            fig2=PlotlyStenbacka(InggroupPart)
            fig3=PlotlyStenbacka(LingroupPart)
            ##           
            
            if select_variable == "Tráfico":
                st.write(TrafgroupPart)
                st.plotly_chart(fig1, use_container_width=True)
            if select_variable == "Ingresos":
                st.write(InggroupPart)
                st.plotly_chart(fig2, use_container_width=True)
            if select_variable == "Líneas":
                st.write(LingroupPart)
                st.plotly_chart(fig3, use_container_width=True)

        if select_indicador == 'Concentración':
            dflistTraf=[];dflistIng=[];dflistLin=[]
            
            for elem in PERIODOS:
                dflistTraf.append(Concentracion(Trafnac,'trafico',elem))
                dflistIng.append(Concentracion(Ingnac,'ingresos',elem))
                dflistLin.append(Concentracion(Linnac,'lineas',elem))
            ConcTraf=pd.concat(dflistTraf).fillna(1.0)
            ConcIng=pd.concat(dflistIng).fillna(1.0)
            ConcLin=pd.concat(dflistLin).fillna(1.0)      
                        
            if select_variable == "Tráfico":
                colsconTraf=ConcTraf.columns.values.tolist()
                conc=st.slider('Seleccionar el número de empresas',1,len(colsconTraf)-1,1,1)
                fig4=PlotlyConcentracion(ConcTraf)
                st.write(ConcTraf.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconTraf[conc]]))
                st.plotly_chart(fig4,use_container_width=True)
            if select_variable == "Ingresos":
                colsconIng=ConcIng.columns.values.tolist()
                conc=st.slider('Seleccione el número de empresas',1,len(colsconIng)-1,1,1)
                fig5=PlotlyConcentracion(ConcIng)
                st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                st.plotly_chart(fig5,use_container_width=True)
            if select_variable == "Líneas":
                colsconLin=ConcLin.columns.values.tolist()
                conc=st.slider('Seleccione el número de empresas',1,len(colsconLin)-1,1,1)
                fig6=PlotlyConcentracion(ConcLin)
                st.write(ConcLin.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconLin[conc]]))
                st.plotly_chart(fig6,use_container_width=True)
    
        if select_indicador == 'IHH':
            PERIODOS=Trafnac['periodo'].unique().tolist()
            for elem in PERIODOS:
                prTr=Trafnac[Trafnac['periodo']==elem]
                prTr.insert(3,'participacion',(prTr['trafico']/prTr['trafico'].sum())*100)
                prTr.insert(4,'IHH',IHH(prTr,'trafico'))
                dfTrafico3.append(prTr.sort_values(by='participacion',ascending=False))
                ##
                prIn=Ingnac[Ingnac['periodo']==elem]
                prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                dfIngresos3.append(prIn.sort_values(by='participacion',ascending=False))
                ##
                prLi=Linnac[Linnac['periodo']==elem]
                prLi.insert(3,'participacion',(prLi['lineas']/prLi['lineas'].sum())*100)
                prLi.insert(4,'IHH',IHH(prLi,'lineas'))
                dfLineas3.append(prLi.sort_values(by='participacion',ascending=False))
            TrafgroupPart3=pd.concat(dfTrafico3)
            InggroupPart3=pd.concat(dfIngresos3)
            LingroupPart3=pd.concat(dfLineas3)
            IHHTraf=TrafgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            IHHLin=LingroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            
            ##Gráficas
            
            fig7 = PlotlyIHH(IHHTraf)   
            fig8 = PlotlyIHH(IHHIng)
            fig9 = PlotlyIHH(IHHLin)  
            
            if select_variable == "Tráfico":
                st.write(TrafgroupPart3)
                st.plotly_chart(fig7,use_container_width=True)
            if select_variable == "Ingresos":
                st.write(InggroupPart3)
                st.plotly_chart(fig8,use_container_width=True)
            if select_variable == "Líneas":
                st.write(LingroupPart3)
                st.plotly_chart(fig9,use_container_width=True)
                
        if select_indicador == 'Linda':
            dflistTraf2=[];dflistIng2=[];dflistLin2=[]
            
            for elem in PERIODOS:
                dflistTraf2.append(Linda(Trafnac,'trafico',elem))
                dflistIng2.append(Linda(Ingnac,'ingresos',elem))
                dflistLin2.append(Linda(Linnac,'lineas',elem))
            LindTraf=pd.concat(dflistTraf2).reset_index().drop('index',axis=1).fillna(0)
            LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).dropna(axis=1) 
            LindLin=pd.concat(dflistLin2).reset_index().drop('index',axis=1).dropna(axis=1)     


            if select_variable == "Tráfico":
                LindconTraf=LindTraf.columns.values.tolist()
                lind=st.slider('Seleccionar nivel',2,len(LindconTraf),2,1)
                fig10=PlotlyLinda(LindTraf)
                st.write(LindTraf.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconTraf[lind-1]]))
                st.plotly_chart(fig10,use_container_width=True)
            if select_variable == "Ingresos":
                LindconIng=LindIng.columns.values.tolist()            
                lind=st.slider('Seleccionar nivel',2,len(LindconIng),2,1)
                fig11=PlotlyLinda(LindIng)
                st.write(LindIng.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconIng[lind-1]]))
                st.plotly_chart(fig11,use_container_width=True)
            if select_variable == "Líneas":
                LindconLin=LindLin.columns.values.tolist()            
                lind=st.slider('Seleccionar nivel',2,len(LindconLin),2,1)
                fig12=PlotlyLinda(LindLin)
                st.write(LindLin.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconLin[lind-1]]))
                st.plotly_chart(fig12,use_container_width=True)                
                            
    if select_dimension == 'Municipal':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda'])
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
        
        st.write('#### Desagregación municipal')
        col1, col2 = st.columns(2)
        with col1:        
            select_variable = st.selectbox('Variable',['Tráfico','Líneas'])  
        MUNICIPIOS=sorted(Trafmuni.codigo.unique().tolist())
        MUNICIPIOSLIN=sorted(Linmuni.codigo.unique().tolist())
        with col2:
            MUNI=st.selectbox('Escoja el municipio', MUNICIPIOS)
        PERIODOSTRAF=Trafmuni[Trafmuni['codigo']==MUNI]['periodo'].unique().tolist()
        PERIODOSLIN=Linmuni[Linmuni['codigo']==MUNI]['periodo'].unique().tolist()   
    
    ## Cálculo de los indicadores 
    
        if select_indicador == 'Stenbacka':                        
            gamma=st.slider('Seleccionar valor gamma',0.0,1.0,0.1)
            
            for periodo in PERIODOSTRAF:
                prTr=Trafmuni[(Trafmuni['codigo']==MUNI)&(Trafmuni['periodo']==periodo)]
                prTr.insert(5,'participacion',Participacion(prTr,'trafico'))
                prTr.insert(6,'stenbacka',Stenbacka(prTr,'trafico',gamma))
                dfTrafico.append(prTr.sort_values(by='participacion',ascending=False))
            TrafgroupPart=pd.concat(dfTrafico)
            
            
            for periodo in PERIODOSLIN:
                prLi=Linmuni[(Linmuni['codigo']==MUNI)&(Linmuni['periodo']==periodo)]
                prLi.insert(5,'participacion',Participacion(prLi,'lineas'))
                prLi.insert(6,'stenbacka',Stenbacka(prLi,'lineas',gamma))
                dfLineas.append(prLi.sort_values(by='participacion',ascending=False))
            LingroupPart=pd.concat(dfLineas)
            

            ##Graficas 
            
            fig1=PlotlyStenbacka(TrafgroupPart)
            fig2=PlotlyStenbacka(LingroupPart)
                  
            if select_variable == "Tráfico":
                st.write(TrafgroupPart)
                st.plotly_chart(fig1,use_container_width=True)
            if select_variable == "Líneas":
                st.write(LingroupPart)
                st.plotly_chart(fig2,use_container_width=True)
   
        if select_indicador == 'Concentración':
            dflistTraf=[];dflistIng=[];dflistLin=[]
            
            for periodo in PERIODOSTRAF:
                prTr=Trafmuni[(Trafmuni['codigo']==MUNI)&(Trafmuni['periodo']==periodo)]
                prLi=Linmuni[(Linmuni['codigo']==MUNI)&(Linmuni['periodo']==periodo)]
                dflistTraf.append(Concentracion(prTr,'trafico',periodo))
                dflistLin.append(Concentracion(prLi,'lineas',periodo))
            ConcTraf=pd.concat(dflistTraf).fillna(1.0).reset_index().drop('index',axis=1)
            ConcLin=pd.concat(dflistLin).fillna(1.0).reset_index().drop('index',axis=1)
            
            
            if select_variable == "Tráfico":
                colsconTraf=ConcTraf.columns.values.tolist()
                value1= len(colsconTraf)-1 if len(colsconTraf)-1 >1 else 2
                conc=st.slider('Seleccione el número de empresas',1,value1,1,1)
                fig3 = PlotlyConcentracion(ConcTraf) 
                st.write(ConcTraf.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconTraf[conc]]))
                st.plotly_chart(fig3,use_container_width=True)   
            if select_variable == "Líneas":
                colsconLin=ConcLin.columns.values.tolist()
                value2= len(colsconLin)-1 if len(colsconLin)-1 >1 else 2
                conc=st.slider('Seleccione el número de empresas',1,value2,1,1)
                fig4 = PlotlyConcentracion(ConcLin)
                st.write(ConcLin.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconLin[conc]]))
                st.plotly_chart(fig4,use_container_width=True)   

        if select_indicador == 'IHH':            
            for periodo in PERIODOSTRAF:
                prTr=Trafmuni[(Trafmuni['codigo']==MUNI)&(Trafmuni['periodo']==periodo)]
                prLi=Linmuni[(Linmuni['codigo']==MUNI)&(Linmuni['periodo']==periodo)]
                prTr.insert(3,'participacion',(prTr['trafico']/prTr['trafico'].sum())*100)
                prTr.insert(4,'IHH',IHH(prTr,'trafico'))
                dfTrafico3.append(prTr.sort_values(by='participacion',ascending=False))
                prLi.insert(3,'participacion',(prLi['lineas']/prLi['lineas'].sum())*100)
                prLi.insert(4,'IHH',IHH(prLi,'lineas'))
                dfLineas3.append(prLi.sort_values(by='participacion',ascending=False))
            TrafgroupPart3=pd.concat(dfTrafico3)
            LingroupPart3=pd.concat(dfLineas3)
            IHHTraf=TrafgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            IHHLin=LingroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()    
            
            fig5=PlotlyIHH(IHHTraf)
            fig6=PlotlyIHH(IHHLin)

            if select_variable == "Tráfico":
                st.write(TrafgroupPart3)
                st.plotly_chart(fig5,use_container_width=True)
            if select_variable == "Líneas":
                st.write(LingroupPart3)
                st.plotly_chart(fig6,use_container_width=True)  

        if select_indicador == 'Linda':
            dflistTraf2=[];dflistLin2=[];datosTraf=[];datosLin=[];nempresaTraf=[];nempresaLin=[];                
            for periodo in PERIODOSTRAF:
                prTr=Trafmuni[(Trafmuni['codigo']==MUNI)&(Trafmuni['periodo']==periodo)]
                nempresaTraf.append(prTr.empresa.nunique())
                dflistTraf2.append(Linda(prTr,'trafico',periodo))
                datosTraf.append(prTr)    
                prLi=Linmuni[(Linmuni['codigo']==MUNI)&(Linmuni['periodo']==periodo)]
                nempresaLin.append(prLi.empresa.nunique())
                dflistLin2.append(Linda(prLi,'lineas',periodo))
                datosLin.append(prLi)
            NemphisTraf=max(nempresaTraf)
            NemphisLin=max(nempresaLin)     
            dTraf=pd.concat(datosTraf).reset_index().drop('index',axis=1)
            LindTraf=pd.concat(dflistTraf2).reset_index().drop('index',axis=1).fillna(np.nan)
            dLin=pd.concat(datosLin).reset_index()
            LindLin=pd.concat(dflistLin2).reset_index().drop('index',axis=1).fillna(np.nan)            
                
            if select_variable == "Tráfico":
                LindconTraf=LindTraf.columns.values.tolist()
                if NemphisTraf==1:
                    st.write("El índice de linda no está definido para éste municipio pues cuenta con una sola empresa")
                    st.write(dTraf)
                elif  NemphisTraf==2:
                    col1, col2 = st.columns([3, 1])
                    fig10=PlotlyLinda2(LindTraf)
                    col1.write("**Datos completos**")                    
                    col1.write(dTraf)  
                    col2.write("**Índice de Linda**")
                    col2.write(LindTraf)
                    st.plotly_chart(fig10,use_container_width=True)        
                else:    
                    lind=st.slider('Seleccionar nivel',2,len(LindconTraf),2,1)
                    fig10=PlotlyLinda(LindTraf)
                    st.write(LindTraf.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconTraf[lind-1]]))
                    with st.expander("Mostrar datos"):
                        st.write(dTraf)                    
                    st.plotly_chart(fig10,use_container_width=True)
 
            if select_variable == "Líneas":
                LindconLin=LindLin.columns.values.tolist()
                if  NemphisLin==1:
                    st.write("El índice de linda no está definido para éste municipio pues cuenta con una sola empresa")
                    st.write(dLin)
                elif  NemphisLin==2:
                    col1, col2 = st.columns([3, 1])
                    fig11=PlotlyLinda2(LindLin)
                    col1.write("**Datos completos**")
                    col1.write(dLin)
                    col2.write("**Índice de Linda**")    
                    col2.write(LindLin)
                    st.plotly_chart(fig11,use_container_width=True)        
                else:
                    lind=st.slider('Seleccionar nivel',2,len(LindconLin),2,1)
                    fig11=PlotlyLinda(LindLin)
                    st.write(LindLin.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconLin[lind-1]]))
                    with st.expander("Mostrar datos"):
                        st.write(dLin)
                    st.plotly_chart(fig11,use_container_width=True)
                
               
    if select_dimension == 'Departamental':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Media entrópica'])
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
        if select_indicador == 'Media entrópica':
            st.write("### Media entrópica")
            st.write(r"""La media entrópica es un índice que tiene los mismos límites superiores e inferiores del IHH/10000 (1/n a 1), donde n es el número de empresas en el mercado. El valor mayor de este índice es 1 y corresponde a una situación de monopolio. En el intermedio el índice tomará valores inferiores al IHH/10000 pero no muy distantes.""")
            with st.expander("Cálculo detallado de la media entrópica"):
                st.write(r""" Para un mercado dividido en submercados, la media entrópica se descompone en tres términos múltiplicativos:
-   **Concentración dentro del submercado:** donde cada submercado trendrá su cálculo de la media entrópica. Este factor, para el mercado en conjunto, tomará valores entre 0 y 1 que representa la concentración dentro del submercado en el conjunto del mercado.

-   **Concentración entre los submercados:** donde cada submercado tendrá su cuota de participación en el mercado total. Para el mercado en conjunto, este factor tomará valores entre 1/n y 1, siendo cercano a 1 en la medida que hayan pocos submercados, en relación al total, con una cuota de participación mayor en el mercado.

-   **Componente de interacción:** Este factor tomará valores mayores que 1. En cada submercado su valor crecerá exponencialmente en la medida que se trate de mercados pequeños atendidos en buena parte por una o pocas empresas grandes en el mercado total. Los valores más altos de este factor para el mercado total puden interpretarse como alertas para hacer un mayor seguimiento a los submercados correspondientes.             

La media entrópica se descompone en tres terminos multiplicativos que resultan de aplicar su definición (ME) a la descomposición del índice de Theil (EI).En el cual, el índice de Theil (Theil, 1967), se representa como la suma de las participaciones del mercado multiplicada cada una por el logaritmo natural de su inverso:

$$IE = \sum_{i=1}^{n} S_{i} ln\frac{1}{S_{i}}$$

**Donde:**

-   $S_{i}$ corresponde a la participación de cada una de las empresas del mercado.

Y por su parte, la media entrópica parte del exponencial del índice de entrópia de Theil ($e^{IE}$), que de acuerdo con Taagepera y Grofman (1981) corresponde a un número efectivo de empresas comparable con el número de empresas equivalentes que se obtienen como el inverso del índice IHH (10000/IHH). Para finalmente, hayar su cálculo a través del inverso del número efectivo de Taagepera y Grofman ($e^{-IE}$) de la siguiente manera:

$$ME = e_{-IE} = \prod_{i=1}^{n} S_{i}^{\frac{S_{i}}{n_{i}}}$$

La media entrópica, al contrario del índice IE, pero en la misma dirección del índice IHH, aumenta cuando crece la concentración, lo cual facilita su interpretación. El límite superior del IE (mínima concentración) es un valor que depende del número de competidores (ln(n); donde n es el número de competidores), mientras que los índices ME e IHH/10000 siempre producen un valor entre cero y uno, correspondiendo para ambos la mínima concentración a 1/n cuando hay n competidores, y tomando ambos el valor de uno (1) para un mercado monopólico (máxima concentración).

#### Descomposición multiplicativa de la media entrópica

La descomposición multiplicativa de la media entrópica se haya de la siguiente manera:

$$ME = ME_{D} * ME_{E} * ME_{I}$$

**Donde:**

-   $ME_{D}$ corresponde al componente de concentración dentro del submercado:

$$ME_{D} = \prod_{j=1}^{p} ME_{D,j}^{w_{j}};$$
$$ME_{D,j} = \prod_{i \in C_{j}}(\frac{S_{ij}}{n_{i}w_{j}})^{(\frac{S_{ij}}{w_{j}})}$$

-   $ME_{E}$ corresponde al componente de concentración entre los submercados:

$$ME_{E} = \prod_{j=1}^{p} W_{j}^{w_{j}}$$

-   $ME_{I}$ corresponde al componente de interacción:

$$ME_{I} = \prod_{j=1}^{p} ME_{I,j}^{w_{j}};$$
$$ME_{I,j} = \prod_{i \in C_{j}}^{n} (\frac{S_{i}}{S_{ij}})^{(\frac{S_{ij}}{w_{j}})}$$

***Donde a su vez de manera general:***

-   $w_{j}$ es:

$$w_{j} = \sum_{i=1}^{n} S_{ij};$$
$$j = 1, 2, ..., p$$

-   $S_{i}$ es:

$$S_{i} = \sum_{j=1}^{p} S_{ij};$$
$$i = 1, 2, ..., n$$

                """)
                
                
        st.write('#### Agregación departamental') 
        col1, col2 = st.columns(2)
        with col1:
            select_variable = st.selectbox('Variable',['Tráfico','Líneas']) 
            
        DEPARTAMENTOSTRAF=sorted(Trafdpto.departamento.unique().tolist())
        DEPARTAMETNOSLIN=sorted(Lindpto.departamento.unique().tolist())
        with col2:
            DPTO=st.selectbox('Escoja el departamento', DEPARTAMENTOSTRAF,5)
        PERIODOSTRAF=Trafdpto[Trafdpto['departamento']==DPTO]['periodo'].unique().tolist()
        PERIODOSLIN=Lindpto[Lindpto['departamento']==DPTO]['periodo'].unique().tolist()
    
    ##Cálculo de los indicadores
    
        if select_indicador == 'Stenbacka':
            gamma=st.slider('Seleccionar valor gamma',0.0,1.0,0.1)            
        
            for periodo in PERIODOSTRAF:
                prTr=Trafdpto[(Trafdpto['departamento']==DPTO)&(Trafdpto['periodo']==periodo)]
                prTr.insert(5,'participacion',Participacion(prTr,'trafico'))
                prTr.insert(6,'stenbacka',Stenbacka(prTr,'trafico',gamma))
                dfTrafico.append(prTr.sort_values(by='participacion',ascending=False))
            TrafgroupPart=pd.concat(dfTrafico) 

            for periodo in PERIODOSLIN:
                prLi=Lindpto[(Lindpto['departamento']==DPTO)&(Lindpto['periodo']==periodo)]
                prLi.insert(5,'participacion',Participacion(prLi,'lineas'))
                prLi.insert(6,'stenbacka',Stenbacka(prLi,'lineas',gamma))
                dfLineas.append(prLi.sort_values(by='participacion',ascending=False))
            LingroupPart=pd.concat(dfLineas)             
          
            ##Graficas 
            
            fig1=PlotlyStenbacka(TrafgroupPart)
            fig2=PlotlyStenbacka(LingroupPart)
            
            if select_variable == "Tráfico":
                st.write(TrafgroupPart)
                st.plotly_chart(fig1,use_container_width=True)
            if select_variable == "Líneas":
                st.write(LingroupPart)
                st.plotly_chart(fig2,use_container_width=True)     
        
        if select_indicador =='Concentración':
            dflistTraf=[];dflistIng=[];dflistLin=[]

            for periodo in PERIODOSTRAF:
                prTr=Trafdpto[(Trafdpto['departamento']==DPTO)&(Trafdpto['periodo']==periodo)]
                prLi=Lindpto[(Lindpto['departamento']==DPTO)&(Lindpto['periodo']==periodo)]
                dflistTraf.append(Concentracion(prTr,'trafico',periodo))
                dflistLin.append(Concentracion(prLi,'lineas',periodo))
            ConcTraf=pd.concat(dflistTraf).fillna(1.0).reset_index().drop('index',axis=1)
            ConcLin=pd.concat(dflistLin).fillna(1.0).reset_index().drop('index',axis=1)
            

            if select_variable == "Tráfico":
                colsconTraf=ConcTraf.columns.values.tolist()
                value1= len(colsconTraf)-1 if len(colsconTraf)-1 >1 else 2 
                conc=st.slider('Seleccionar número de expresas ',1,value1,1,1)
                fig3 = PlotlyConcentracion(ConcTraf) 
                st.write(ConcTraf.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconTraf[conc]]))
                st.plotly_chart(fig3,use_container_width=True)   
            if select_variable == "Líneas":
                colsconLin=ConcLin.columns.values.tolist()
                value2= len(colsconLin)-1 if len(colsconLin)-1 >1 else 2 
                conc=st.slider('Seleccionar número de expresas ',1,value2,1,1)
                fig4 = PlotlyConcentracion(ConcLin)
                st.write(ConcLin.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconLin[conc]]))
                st.plotly_chart(fig4,use_container_width=True)   

        if select_indicador == 'IHH':
            
            for periodo in PERIODOSTRAF:
                prTr=Trafdpto[(Trafdpto['departamento']==DPTO)&(Trafdpto['periodo']==periodo)]
                prLi=Lindpto[(Lindpto['departamento']==DPTO)&(Lindpto['periodo']==periodo)]
                prTr.insert(3,'participacion',(prTr['trafico']/prTr['trafico'].sum())*100)
                prTr.insert(4,'IHH',IHH(prTr,'trafico'))
                dfTrafico3.append(prTr.sort_values(by='participacion',ascending=False))
                prLi.insert(3,'participacion',(prLi['lineas']/prLi['lineas'].sum())*100)
                prLi.insert(4,'IHH',IHH(prLi,'lineas'))
                dfLineas3.append(prLi.sort_values(by='participacion',ascending=False))
            TrafgroupPart3=pd.concat(dfTrafico3)
            LingroupPart3=pd.concat(dfLineas3)
            IHHTraf=TrafgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            IHHLin=LingroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()    
            
            fig5=PlotlyIHH(IHHTraf)
            fig6=PlotlyIHH(IHHLin)

            if select_variable == "Tráfico":
                st.write(TrafgroupPart3)
                st.plotly_chart(fig5,use_container_width=True)
            if select_variable == "Líneas":
                st.write(LingroupPart3)
                st.plotly_chart(fig6,use_container_width=True)    
                
        if select_indicador == 'Linda':
            dflistTraf2=[];dflistLin2=[];datosTraf=[];datosLin=[];nempresaTraf=[];nempresaLin=[];       
            for periodo in PERIODOSTRAF:              
                prTr=Trafdpto[(Trafdpto['departamento']==DPTO)&(Trafdpto['periodo']==periodo)]
                prLi=Lindpto[(Lindpto['departamento']==DPTO)&(Lindpto['periodo']==periodo)]
                nempresaTraf.append(prTr.empresa.nunique())
                dflistTraf2.append(Linda(prTr,'trafico',periodo))
                datosTraf.append(prTr)
                nempresaLin.append(prLi.empresa.nunique())
                dflistLin2.append(Linda(prLi,'lineas',periodo))
                datosLin.append(prLi)
            NemphisTraf=max(nempresaTraf)
            NemphisLin=max(nempresaLin)     
            dTraf=pd.concat(datosTraf).reset_index().drop('index',axis=1)
            LindTraf=pd.concat(dflistTraf2).reset_index().drop('index',axis=1).fillna(np.nan)
            dLin=pd.concat(datosLin).reset_index()
            LindLin=pd.concat(dflistLin2).reset_index().drop('index',axis=1).fillna(np.nan)   

            if select_variable == "Tráfico":
                LindconTraf=LindTraf.columns.values.tolist()
                if NemphisTraf==1:
                    st.write("El índice de linda no está definido para éste departamento pues cuenta con una sola empresa")
                    st.write(dTraf)
                elif  NemphisTraf==2:
                    col1, col2 = st.columns([3, 1])
                    fig10=PlotlyLinda2(LindTraf)
                    col1.write("**Datos completos**")                    
                    col1.write(dTraf)  
                    col2.write("**Índice de Linda**")
                    col2.write(LindTraf)
                    st.plotly_chart(fig10,use_container_width=True)        
                else:    
                    lind=st.slider('Seleccionar nivel',2,len(LindconTraf),2,1)
                    fig10=PlotlyLinda(LindTraf)
                    st.write(LindTraf.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconTraf[lind-1]]))
                    with st.expander("Mostrar datos"):
                        st.write(dTraf)                    
                    st.plotly_chart(fig10,use_container_width=True)
 
            if select_variable == "Líneas":
                LindconLin=LindLin.columns.values.tolist()
                if  NemphisLin==1:
                    st.write("El índice de linda no está definido para éste departamento pues cuenta con una sola empresa")
                    st.write(dLin)
                elif  NemphisLin==2:
                    col1, col2 = st.columns([3, 1])
                    fig11=PlotlyLinda2(LindLin)
                    col1.write("**Datos completos**")
                    col1.write(dLin)
                    col2.write("**Índice de Linda**")    
                    col2.write(LindLin)
                    st.plotly_chart(fig11,use_container_width=True)        
                else:
                    lind=st.slider('Seleccionar nivel',2,len(LindconLin),2,1)
                    fig11=PlotlyLinda(LindLin)
                    st.write(LindLin.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconLin[lind-1]]))
                    with st.expander("Mostrar datos"):
                        st.write(dLin)
                    st.plotly_chart(fig11,use_container_width=True)            
            
        if select_indicador == 'Media entrópica':

            for periodo in PERIODOSTRAF:
                prTr=Trafico[(Trafico['departamento']==DPTO)&(Trafico['periodo']==periodo)]
                prTr.insert(4,'media entropica',MediaEntropica(prTr,'trafico')[0])
                dfTrafico.append(prTr)
                prLi=Lineas[(Lineas['departamento']==DPTO)&(Lineas['periodo']==periodo)]
                prLi.insert(4,'media entropica',MediaEntropica(prLi,'lineas')[0])
                dfLineas.append(prLi)
            TrafgroupPart=pd.concat(dfTrafico)
            MEDIAENTROPICATRAF=TrafgroupPart.groupby(['periodo'])['media entropica'].mean().reset_index()    
            LingroupPart=pd.concat(dfLineas)
            MEDIAENTROPICALIN=LingroupPart.groupby(['periodo'])['media entropica'].mean().reset_index()            
            #Graficas
            
            fig7=PlotlyMEntropica(MEDIAENTROPICATRAF)
            fig8=PlotlyMEntropica(MEDIAENTROPICALIN)

            
            if select_variable == "Tráfico":
                periodoME=st.selectbox('Escoja un periodo para calcular la media entrópica', PERIODOSTRAF,len(PERIODOSTRAF)-1)
                MEperiodTableTraf=MediaEntropica(Trafico[(Trafico['departamento']==DPTO)&(Trafico['periodo']==periodoME)],'trafico')[1] 
                st.write(r"""##### <center>Visualización de la evolución de la media entrópica en el departamento seleccionado</center>""",unsafe_allow_html=True)
                st.plotly_chart(fig7,use_container_width=True)
                dfMap=[];
                for departamento in DEPARTAMENTOSTRAF:
                    prTr=Trafico[(Trafico['departamento']==departamento)&(Trafico['periodo']==periodoME)]
                    prTr.insert(4,'media entropica',MediaEntropica(prTr,'trafico')[0])
                    prTr2=prTr.groupby(['id_departamento','departamento'])['media entropica'].mean().reset_index()
                    dfMap.append(prTr2)
                TraMap=pd.concat(dfMap).reset_index().drop('index',axis=1)
                colsME=['SIJ','SI','WJ','MED','MEE','MEI','Media entropica'] 
                st.write(MEperiodTableTraf.reset_index(drop=True).style.apply(f, axis=0, subset=colsME))
                departamentos_df=gdf.merge(TraMap, on='id_departamento')
                departamentos_df['media entropica']=departamentos_df['media entropica'].round(4)
                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'media entropica'],
                    key_on='feature.properties.DPTO',
                    fill_color='Greens', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Media entrópica',
                    bins=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                    smooth_factor=0).add_to(colombia_map)
                # Adicionar nombres del departamento
                style_function = "font-size: 15px; font-weight: bold"
                choropleth.geojson.add_child(
                    folium.features.GeoJsonTooltip(['NOMBRE_DPT'], style=style_function, labels=False))
                folium.LayerControl().add_to(colombia_map)

                #Adicionar valores velocidad
                style_function = lambda x: {'fillColor': '#ffffff', 
                                            'color':'#000000', 
                                            'fillOpacity': 0.1, 
                                            'weight': 0.1}
                highlight_function = lambda x: {'fillColor': '#000000', 
                                                'color':'#000000', 
                                                'fillOpacity': 0.50, 
                                                'weight': 0.1}
                NIL = folium.features.GeoJson(
                    data = departamentos_df,
                    style_function=style_function, 
                    control=False,
                    highlight_function=highlight_function, 
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['id_departamento','departamento_y','media entropica'],
                        aliases=['ID Departamento','Departamento','Media entrópica'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                MunicipiosME=MEperiodTableTraf.groupby(['municipio'])['WJ'].mean().reset_index()
                MunicipiosME=MunicipiosME[MunicipiosME.WJ!=0]
                MunicipiosME.WJ=MunicipiosME.WJ.round(7)
                
                
                fig9=PlotlyMentropicaTorta(MunicipiosME)
                
                col1, col2= st.columns(2)
                with col1:
                    st.write(r"""###### <center>Visualización de la media entrópica en todos los departamentos y en el periodo seleccionado</center>""",unsafe_allow_html=True)
                    folium_static(colombia_map,width=480)    
                with col2:
                    st.write(r"""###### <center>Visualización de la participación de los municipios dentro del departamento seleccionado</center>""",unsafe_allow_html=True)                
                    st.plotly_chart(fig9,use_container_width=True)
                
            if select_variable == "Líneas":
                periodoME2=st.selectbox('Escoja un periodo para calcular la media entrópica', PERIODOSLIN,len(PERIODOSLIN)-1)
                MEperiodTableLin=MediaEntropica(Lineas[(Lineas['departamento']==DPTO)&(Lineas['periodo']==periodoME2)],'lineas')[1] 
                st.write(r"""##### <center>Visualización de la evolución de la media entrópica en el departamento seleccionado</center>""",unsafe_allow_html=True)
                st.plotly_chart(fig8,use_container_width=True)
                dfMap2=[];
                for departamento in DEPARTAMENTOSTRAF:
                    prLi=Lineas[(Lineas['departamento']==departamento)&(Lineas['periodo']==periodoME2)]
                    prLi.insert(4,'media entropica',MediaEntropica(prLi,'lineas')[0])
                    prLi2=prLi.groupby(['id_departamento','departamento'])['media entropica'].mean().reset_index()
                    dfMap2.append(prLi2)
                LinMap=pd.concat(dfMap2)
                LinMap=LinMap.reset_index().drop('index',axis=1)
                colsME=['SIJ','SI','WJ','MED','MEE','MEI','Media entropica'] 
                st.write(MEperiodTableLin.reset_index(drop=True).style.apply(f, axis=0, subset=colsME))
                st.write(r"""##### <center>Visualización de la media entrópica en el periodo seleccionado</center>""",unsafe_allow_html=True)
                departamentos_df2=gdf.merge(LinMap, on='id_departamento')
                departamentos_df2['media entropica']=departamentos_df2['media entropica'].round(3)
                colombia_map2 = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map2)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df2,
                    columns=['id_departamento', 'media entropica'],
                    key_on='feature.properties.DPTO',
                    fill_color='Greens', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Media entrópica',
                    bins=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                    smooth_factor=0).add_to(colombia_map2)
                # Adicionar nombres del departamento
                style_function = "font-size: 15px; font-weight: bold"
                choropleth.geojson.add_child(
                    folium.features.GeoJsonTooltip(['NOMBRE_DPT'], style=style_function, labels=False))
                folium.LayerControl().add_to(colombia_map2)

                #Adicionar valores velocidad
                style_function = lambda x: {'fillColor': '#ffffff', 
                                            'color':'#000000', 
                                            'fillOpacity': 0.1, 
                                            'weight': 0.1}
                highlight_function = lambda x: {'fillColor': '#000000', 
                                                'color':'#000000', 
                                                'fillOpacity': 0.50, 
                                                'weight': 0.1}
                NIL = folium.features.GeoJson(
                    data = departamentos_df2,
                    style_function=style_function, 
                    control=False,
                    highlight_function=highlight_function, 
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['id_departamento','departamento_y','media entropica'],
                        aliases=['ID Departamento','Departamento','Media entrópica'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map2.add_child(NIL)
                colombia_map2.keep_in_front(NIL)
                MunicipiosME2=MEperiodTableLin.groupby(['municipio'])['WJ'].mean().reset_index()
                MunicipiosME2=MunicipiosME2[MunicipiosME2.WJ!=0]
                MunicipiosME2.WJ=MunicipiosME2.WJ.round(7)
                
                fig10=PlotlyMentropicaTorta(MunicipiosME2)
                
                col1, col2 ,= st.columns2
                with col1:
                    st.write(r"""###### <center>Visualización de la media entrópica en todos los departamentos y en el periodo seleccionado</center>""",unsafe_allow_html=True)
                    folium_static(colombia_map2,width=480)
                with col2:       
                    st.write(r"""###### <center>Visualización de la participación de los municipios dentro del departamento seleccionado</center>""",unsafe_allow_html=True)
                    st.plotly_chart(fig10,use_container_width=True)                
 
            

if select_mercado == "Internet fijo":
    st.write("# Internet fijo")
    
if select_mercado == "Televisión por suscripción":
    st.write("# Televisión por suscripción")    
