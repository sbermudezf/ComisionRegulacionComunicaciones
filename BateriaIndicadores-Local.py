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
    return dfAgg['Media entropica'].unique().tolist()[0],dfAgg
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
    h1{ background: #F08080;
    text-align: center;
    padding: 15px;
    font-family: sans-serif;
    color: black;}
    .barra-superior{top: 0;
    position: fixed;
    background-color: #3366cc;
    width: 100%;
    color:white;
    z-index: 999;
    height: 100px;
    left: 0px;
    text-align: center;
    padding: 15px;
    font-size: 36px;
    font-weight: 700;
    }
    .css-1lcbmhc > div{margin-top:100px;}
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
    </style>""", unsafe_allow_html=True)  
st.markdown("""<div class="barra-superior">
<img src="https://www.postdata.gov.co/sites/all/themes/nuboot_radix/logo-crc-blanco.png"    
 alt="0" style="height: 70px; left:10px; position:fixed;">
Batería de indicadores para el análisis de competencia
</div>""",unsafe_allow_html=True)
#st.sidebar.image(LogoComision2, use_column_width=True)
indicacionesuso="""
<br></br>
 * Utilice el menú de la izquierda para seleccionar el indicador
 * Las gráficas y los datos apareceran debajo
"""
st.sidebar.markdown("""<b>Seleccione el indicador a calcular</b>""", unsafe_allow_html=True)

select_mercado = st.sidebar.selectbox('Mercado',
                                    ['Telefonía local', 'Internet fijo','Televisión por suscripción'])

select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Media entrópica'])
                              
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
    st.markdown(indicacionesuso,unsafe_allow_html=True)
    #st.write('## TELEFONÍA LOCAL')    
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
    
    select_dimension=st.sidebar.selectbox('Dimensión',['Departamental','Municipal','Nacional','Clusters'])
    
    if select_dimension == 'Nacional':
        st.write('#### Agregación nacional') 
        select_variable = st.selectbox('Variable',['Tráfico', 'Ingresos','Líneas']) 
        if select_indicador == 'Stenbacka':
            st.write("### Indice de Stenbacka")
            st.markdown("Este indicador trata de calcular un umbral a partir del cual la entidad líder podría disfrutar de poder de mercado. Para esto parte de la participación de mercado de la empresa líder y de la empresa con la segunda participación más grande para calcular un umbral de cuota de mercado después de la cual la empresa líder presentaría posición de dominio")
            st.latex(r'''S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]''')
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
            st.write("### Razón de concentración")
            st.markdown("La razón de concentración de n empresas se calcula como la participación de mercado acumulada de las compañías líderes en el mercado relevante")
            st.latex(r''' CR_{n}=S_1+S_2+S_3+...+S_n=\sum_{i=1}^{n}S_{i}''')
            dflistTraf=[];dflistIng=[];dflistLin=[]
            
            for elem in PERIODOS:
                dflistTraf.append(Concentracion(Trafnac,'trafico',elem))
                dflistIng.append(Concentracion(Ingnac,'ingresos',elem))
                dflistLin.append(Concentracion(Linnac,'lineas',elem))
            ConcTraf=pd.concat(dflistTraf).fillna(1.0)
            ConcIng=pd.concat(dflistIng).fillna(1.0)
            ConcLin=pd.concat(dflistLin).fillna(1.0)      
            conc=st.slider('Seleccionar nivel concentración ',1,19,1,1)
            
            #Gráficas
            fig4=PlotlyConcentracion(ConcTraf)
            fig5=PlotlyConcentracion(ConcIng) 
            fig6=PlotlyConcentracion(ConcLin)
            
            if select_variable == "Tráfico":
                colsconTraf=ConcTraf.columns.values.tolist()
                st.write(ConcTraf.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconTraf[conc]]))
                st.plotly_chart(fig4,use_container_width=True)
            if select_variable == "Ingresos":
                colsconIng=ConcIng.columns.values.tolist()
                st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                st.plotly_chart(fig5,use_container_width=True)
            if select_variable == "Líneas":
                colsconLin=ConcLin.columns.values.tolist()
                st.write(ConcLin.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconLin[conc]]))
                st.plotly_chart(fig6,use_container_width=True)
    
        if select_indicador == 'IHH':
            st.write("### Indice de Herfindahl-Hirschman")
            st.markdown("El Índice Herfindahl-Hirschman (IHH) ha sido uno de los más usados para medir concentraciones de mercados, siendo utilizado desde su planteamiento en 1982 por el Departamento de Justicia de Estados Unidos. Se calcula de la siguiente manera:")
            st.latex(r'''IHH=\sum_{i=1}^{n}S_{i}^{2}''')
            st.write("donde *Si* es la participación de mercado de cada una de las empresas del mercado a analizar, dado en unidades porcentuales.")
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
                            
    if select_dimension == 'Municipal':
        st.write('#### Desagregación municipal')
        select_variable = st.selectbox('Variable',['Tráfico','Líneas'])  
        MUNICIPIOS=sorted(Trafmuni.codigo.unique().tolist())
        MUNICIPIOSLIN=sorted(Linmuni.codigo.unique().tolist())
        MUNI=st.selectbox('Escoja el municipio', MUNICIPIOS)
        PERIODOSTRAF=Trafmuni[Trafmuni['codigo']==MUNI]['periodo'].unique().tolist()
        PERIODOSLIN=Linmuni[Linmuni['codigo']==MUNI]['periodo'].unique().tolist()        
        if select_indicador == 'Stenbacka':
            st.write("### Indice de Stenbacka")
            st.markdown("Este indicador trata de calcular un umbral a partir del cual la entidad líder podría disfrutar de poder de mercado. Para esto parte de la participación de mercado de la empresa líder y de la empresa con la segunda participación más grande para calcular un umbral de cuota de mercado después de la cual la empresa líder presentaría posición de dominio")
            st.latex(r'''S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]''')

            
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
            st.write("### Razón de concentración")
            st.markdown("La razón de concentración de n empresas se calcula como la participación de mercado acumulada de las compañías líderes en el mercado relevante")
            st.latex(r''' CR_{n}=S_1+S_2+S_3+...+S_n=\sum_{i=1}^{n}S_{i}''')
            dflistTraf=[];dflistIng=[];dflistLin=[]
            
            for periodo in PERIODOSTRAF:
                prTr=Trafmuni[(Trafmuni['codigo']==MUNI)&(Trafmuni['periodo']==periodo)]
                prLi=Linmuni[(Linmuni['codigo']==MUNI)&(Linmuni['periodo']==periodo)]
                dflistTraf.append(Concentracion(prTr,'trafico',periodo))
                dflistLin.append(Concentracion(prLi,'lineas',periodo))
            ConcTraf=pd.concat(dflistTraf).fillna(1.0).reset_index().drop('index',axis=1)
            ConcLin=pd.concat(dflistLin).fillna(1.0).reset_index().drop('index',axis=1)
            conc=st.slider('Seleccionar nivel concentración ',1,19,1,1)
            
            #Gráficas
            fig3 = PlotlyConcentracion(ConcTraf) 
            fig4 = PlotlyConcentracion(ConcLin)
            
            if select_variable == "Tráfico":
                colsconTraf=ConcTraf.columns.values.tolist()
                st.write(ConcTraf.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconTraf[conc]]))
                st.plotly_chart(fig3,use_container_width=True)   
            if select_variable == "Líneas":
                colsconLin=ConcLin.columns.values.tolist()
                st.write(ConcLin.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconLin[conc]]))
                st.plotly_chart(fig4,use_container_width=True)   

        if select_indicador == 'IHH':
            st.write("### Indice de Herfindahl-Hirschman")
            st.markdown("El Índice Herfindahl-Hirschman (IHH) ha sido uno de los más usados para medir concentraciones de mercados, siendo utilizado desde su planteamiento en 1982 por el Departamento de Justicia de Estados Unidos. Se calcula de la siguiente manera:")
            st.latex(r'''IHH=\sum_{i=1}^{n}S_{i}^{2}''')
            st.write("donde *Si* es la participación de mercado de cada una de las empresas del mercado a analizar, dado en unidades porcentuales.")

            
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
    
    
    if select_dimension == 'Departamental':
        st.write('#### Agregación departamental') 
        select_variable = st.selectbox('Variable',['Tráfico','Líneas']) 
        DEPARTAMENTOSTRAF=sorted(Trafdpto.departamento.unique().tolist())
        DEPARTAMETNOSLIN=sorted(Lindpto.departamento.unique().tolist())
        DPTO=st.selectbox('Escoja el departamento', DEPARTAMENTOSTRAF)
        PERIODOSTRAF=Trafdpto[Trafdpto['departamento']==DPTO]['periodo'].unique().tolist()
        PERIODOSLIN=Lindpto[Lindpto['departamento']==DPTO]['periodo'].unique().tolist()
        
        if select_indicador == 'Stenbacka':
            st.write("### Indice de Stenbacka")
            st.markdown("Este indicador trata de calcular un umbral a partir del cual la entidad líder podría disfrutar de poder de mercado. Para esto parte de la participación de mercado de la empresa líder y de la empresa con la segunda participación más grande para calcular un umbral de cuota de mercado después de la cual la empresa líder presentaría posición de dominio")
            st.latex(r'''S^{D}=\frac{1}{2}\left[1-\gamma(S_{1}^{2}-S_{2}^{2})\right]''')

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
            st.write("### Razón de concentración")
            st.markdown("La razón de concentración de n empresas se calcula como la participación de mercado acumulada de las compañías líderes en el mercado relevante")
            st.latex(r''' CR_{n}=S_1+S_2+S_3+...+S_n=\sum_{i=1}^{n}S_{i}''')
            dflistTraf=[];dflistIng=[];dflistLin=[]

            for periodo in PERIODOSTRAF:
                prTr=Trafdpto[(Trafdpto['departamento']==DPTO)&(Trafdpto['periodo']==periodo)]
                prLi=Lindpto[(Lindpto['departamento']==DPTO)&(Lindpto['periodo']==periodo)]
                dflistTraf.append(Concentracion(prTr,'trafico',periodo))
                dflistLin.append(Concentracion(prLi,'lineas',periodo))
            ConcTraf=pd.concat(dflistTraf).fillna(1.0).reset_index().drop('index',axis=1)
            ConcLin=pd.concat(dflistLin).fillna(1.0).reset_index().drop('index',axis=1)
            conc=st.slider('Seleccionar nivel concentración ',1,19,1,1)
            
            #Gráficas
            fig3 = PlotlyConcentracion(ConcTraf) 
            fig4 = PlotlyConcentracion(ConcLin)
            
            if select_variable == "Tráfico":
                colsconTraf=ConcTraf.columns.values.tolist()
                st.write(ConcTraf.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconTraf[conc]]))
                st.plotly_chart(fig3,use_container_width=True)   
            if select_variable == "Líneas":
                colsconLin=ConcLin.columns.values.tolist()
                st.write(ConcLin.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconLin[conc]]))
                st.plotly_chart(fig4,use_container_width=True)   

        if select_indicador == 'IHH':
            st.write("### Indice de Herfindahl-Hirschman")
            st.markdown("El Índice Herfindahl-Hirschman (IHH) ha sido uno de los más usados para medir concentraciones de mercados, siendo utilizado desde su planteamiento en 1982 por el Departamento de Justicia de Estados Unidos. Se calcula de la siguiente manera:")
            st.latex(r'''IHH=\sum_{i=1}^{n}S_{i}^{2}''')
            st.write("donde *Si* es la participación de mercado de cada una de las empresas del mercado a analizar, dado en unidades porcentuales.")

            
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
                
        if select_indicador == 'Media entrópica':
            st.write("### Media entrópica")
            st.markdown("La media entrópica se descompone en tres términos multiplicativos que resultan de aplicar la definición de la misma (𝑀𝐸=𝑒−𝐼𝐸) a la descomposición del índice de Theil.")
            st.latex(r'''ME=ME_{D}\times ME_{E}\times ME_{I}\;(1)''')
            st.write("Donde los términos de la ecuación anterior corresponden a las siguientes definiciones:")
            st.latex(r'''ME_{D}=\prod_{j=1}^{p}(ME_{D,j})^{w_{j}}\;(2)\;\; ; \;\;ME_{D,j}=\prod_{i\in C_{j}}\left(\frac{S_{ij}}{n_i w_j}\right)^{(S_{ij}/w_j)}\;(3)\;\; ; \;\;ME_{E}=\prod_{j=1}^{p}w_{j}^{w_j}\;(4)''')               
            st.latex(r'''ME_{I}=\prod_{j=1}^{p}(ME_{I,j})^{w_{j}}\;(5)\;\; ; \;\;ME_{I,j}=\prod_{i\in C_{j}}\left(\frac{S_{i}}{S_{ij}}\right)^{(S_{ij}/w_j)}\;(6)''')
            st.latex(r'''w_{j}=\sum_{i=1}^{n}S_{ij}\;,\;j=1,2,...,p\;(7)\;\;\;\;;\;\;\;\;S_{i}=\sum_{j=1}^{p}S_{ij}\;,\;i=1,2,...,n\;(8)''')

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
                st.write("##### Visualización de la evolución de la media entrópica en el departamento seleccionado")
                st.plotly_chart(fig7,use_container_width=True)
                periodoME=st.selectbox('Escoja un periodo para calcular la media entrópica', PERIODOSTRAF)
                dfMap=[];
                for departamento in DEPARTAMENTOSTRAF:
                    prTr=Trafico[(Trafico['departamento']==departamento)&(Trafico['periodo']==periodoME)]
                    prTr.insert(4,'media entropica',MediaEntropica(prTr,'trafico')[0])
                    prTr2=prTr.groupby(['id_departamento','departamento'])['media entropica'].mean().reset_index()
                    dfMap.append(prTr2)
                TrafMap=pd.concat(dfMap)
                TrafMap=TrafMap.reset_index().drop('index',axis=1)
                MEperiodTableTraf=MediaEntropica(Trafico[(Trafico['departamento']==DPTO)&(Trafico['periodo']==periodoME)],'trafico')[1] 
                colsME=['SIJ','SI','WJ','MED','MEE','MEI','Media entropica'] 
                st.write(MEperiodTableTraf.reset_index(drop=True).style.apply(f, axis=0, subset=colsME))
                st.write("##### Visualización de la media entrópica en el periodo seleccionado")
                departamentos_df=gdf.merge(TrafMap, on='id_departamento')
                departamentos_df['media entropica']=departamentos_df['media entropica'].round(3)
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

                folium_static(colombia_map)                
                
            if select_variable == "Líneas":
                periodoME=st.selectbox('Escoja un periodo para calcular la media entrópica', PERIODOSLIN)
                MEperiodTableLin=MediaEntropica(Lineas[(Lineas['departamento']==DPTO)&(Lineas['periodo']==periodoME)],'lineas')[1] 
                st.write("##### Visualización de la evolución de la media entrópica en el departamento seleccionado")
                st.plotly_chart(fig8,use_container_width=True)
                #periodoME=st.selectbox('Escoja un periodo para calcular la media entrópica', PERIODOSLIN)
                dfMap2=[];
                for departamento in DEPARTAMENTOSTRAF:
                    prLi=Lineas[(Lineas['departamento']==departamento)&(Lineas['periodo']==periodoME)]
                    prLi.insert(4,'media entropica',MediaEntropica(prLi,'lineas')[0])
                    prLi2=prLi.groupby(['id_departamento','departamento'])['media entropica'].mean().reset_index()
                    dfMap2.append(prLi2)
                LinMap=pd.concat(dfMap2)
                LinMap=LinMap.reset_index().drop('index',axis=1)
                colsME=['SIJ','SI','WJ','MED','MEE','MEI','Media entropica'] 
                st.write(MEperiodTableLin.reset_index(drop=True).style.apply(f, axis=0, subset=colsME))
                st.write("##### Visualización de la media entrópica en el periodo seleccionado")
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

                folium_static(colombia_map2)                
                