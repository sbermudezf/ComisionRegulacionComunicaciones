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

Colombian_MUNI=json.load(open("co_2018_MGN_MPIO_POLITICO.geojson", 'r'))
gdf2 = gpd.read_file("co_2018_MGN_MPIO_POLITICO.geojson")
gdf2=gdf2.rename(columns={'MPIO_CNMBR':'municipio','MPIO_CCNCT':'id_municipio'})
#gdf2.id_municipio=gdf2.id_municipio.str.lstrip('0')
gdf2.insert(1,'codigo',gdf2['municipio']+' - '+gdf2['id_municipio'])

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
@st.cache
def Dominancia(df,column):
    part=(df[column]/df[column].sum())*100
    IHH=round(sum([elem**2 for elem in part]),2)
    dom=round(sum([elem**4/IHH**2 for elem in part]),3)
    return dom    
@st.cache
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

def PlotlyIHH(df):    
    fig = make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=df['IHH'],
                         hovertemplate =
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>IHH</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=None,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="Índice de Herfindahl Hirschman", row=1, col=1)
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

def PlotlyDominancia(df):    
    fig = make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=df['Dominancia'],
                         hovertemplate =
        '<br><b>Periodo</b>: %{x}<br>'+                         
        '<br><b>Dominancia</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=None,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="Dominancia", row=1, col=1)
    fig.update_layout(height=550,title="<b> Índice de dominancia</b>",title_x=0.5,legend_title=None,font=dict(family="Helvetica",color=" black"))
    fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-90,showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_traces(marker_color='rgb(204,102,0)', marker_line_color='rgb(102,51,0)',
                      marker_line_width=1.5, opacity=0.4)
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
    fig.update_yaxes(showgrid=True,gridwidth=1,range=[0,flatten(df.iloc[:, [lind-1]].values)],gridcolor='rgba(220, 220, 220, 0.4)',type="linear",rangemode="tozero")
    fig.update_traces(marker_color='rgb(127,0,255)', marker_line_color='rgb(51,0,102)',
                  marker_line_width=1.5, opacity=0.4)
    return fig
def PlotlyLinda2(df):
    fig= make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=df['periodo'], y=df['Linda (2)'],hovertemplate =
    '<br><b>Periodo</b>: %{x}<br>'+                         
    '<br><b>Linda</b>: %{y:.4f}<br>',name=''))
    fig.update_xaxes(tickangle=0, tickfont=dict(family='Helvetica', color='black', size=12),title_text=N,row=1, col=1)
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='black', size=14),titlefont_size=14, title_text="Linda", row=1, col=1)
    fig.update_layout(height=550,title="<b> Índice de Linda por periodo</b>",title_x=0.5,legend_title=None,font=dict(family="Helvetica",color=" black"))
    fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickangle=-90,showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220, 220, 220, 0.4)',type="log", tickvals=[0.5,0.7,0.8,0.9,1.0,1.5,2.0,3.0,5.0,10,50,100,250,500,750,1000])
    fig.update_traces(marker_color='rgb(127,0,255)', marker_line_color='rgb(51,0,102)',
                  marker_line_width=1.5, opacity=0.4)        
    return fig                
LogoComision="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAkFBMVEX/////K2b/AFf/J2T/AFb/ImL/IGH/G1//Fl3/BVn/EVv//f7/mK//9/n/1+D/7fH/PXH/w9D/0tz/aY3/tsb/qr3/4uj/iKP/6u//y9b/RHX/5ev/ssP/8/b/dZX/NWz/UX3/hqL/XYX/obb/fJv/u8r/VH//XIT/gJ3/lKz/Snn/l6//ZYr/bpH/dpb/AEtCvlPnAAAR2UlEQVR4nO1d2XrqPK9eiXEcO8xjoUxlLHzQff93tzFQCrFsy0po1/qfvkc9KIkVy5ol//nzi1/84he/+MXfgUZ/2Bovd7vBBbvqsttqv05+elll4GXYGxxmSkqlUiFEcsHpr1QpqdLmcTdu/7OEvqx3WxGrNOEssoHxE6mVqLMc/mtkvo6nkVSCW0nL06lk8239r1CZDQeRTBP7xlnITJQcVes/vXovauujUsHU3agUkr0Pf5oGF4Yn8pCc6dhKPvhLd/J1J4qS90mknC3/vjPZ2saCypwAkamc/lUbmfWicrbvDoncr3+ark/Udiotb/u+wFQ0/mnaNGoDJZ5A3pVG1vtp+rLq8+g705hG3R8lcCzQ9J0Ml7MxerLj+BknY1Vbq4nvd6r5cxpy2FSI86dtT1nh8+Outx7WXye1WnZGrdbot1u9dx+JEZOL1x+hb9KRXvq0wck6u3W9Zn3MUPk/Eo9330jYJ3rS8/FPJli6rQ4bnucsUXwuou9m1de589OfbK/KZlnPEE9aebn08sR4aueDJ2AZOxT8iTzx0cKuZ49VpUnyfds42Tg2kCsR4h5kuC28bOP782h6QCu1biATlUMLw5s3vEg0hafTOOs/i6h7vMU2vjqZWcE+AUaU3m/j8+24yT61vJ3LTSv8eb1Akyj+KJ+mB9RtsRde6ZDcHaQo/YIYPdV1HFdgDuXySDwh82CvhKdP9BwHMfhOFh/IEiDoGF5fV3ma43gEl8PUiP5Rg0TpDfGyRKq+kM1BoSBYEfcmTJTeIN9KI+sLtREkE1jlLUj95TG2SWYP1LQsum6ozSAhmjaDGLRRX/d279PtfnbGaPOBttmMNx9KJrABEcjkf9jfv7SW070652cSzm5wpDR8EItSCZxEAIFYG6q97OgkBjkS/h0kgiwqV4hf9pcLnaF5RiguEuUxatY0CWTKr5Tag0hi808UpKWJm7kpRZPZi+dH9QGTZTNmHqokpXEw9aDquH9S6zVliUF+K2S1DALfTZXlCQz1358TBAdQhgHXM+wqVnFaMe2FL0ZVJuLCZviwYhAoXUGK9lw+UbaYYKkvmOeBaRkzl/NS31oDAM8CbxajsJlfMEvs8efG8Xv37wJRSGdM82KUJXYtUY29OQienJMX6lxd4ypDCYEskJ8a53nUsYPtmctNYEmqYjE6rKrLcWs4HLa6vepqMYsJRRsAiWT/+zUvZew7mK3sB5CnUm0G3TogErJ6d9CU9OKN67JmVArzh5BZP1Y7soTMdPy703NL9EnrPSpmHwhiAG6QZzvZtvznzrKBiYwGbZSHXN9FRaSUJMQxTy/N82hsecwEztKwNH23fRIIwyN9I5mgpG1muddJS/inDboPXI66ofGNSZVTrb3EYyhDGOROVmpxB8EQKo+3Idt3QzZmRBrD+bSfC40mG/j/3oBwIJNburU45qTgFGOhHJMLETEGM3oHOIIFSwuyqqJY7mIQ9ppxbuUVcFOyjakkeBET44JGh2LdVoL0fpY7DfCqs735seWhjMTJ0KZfHeCWcwQjJ2ZgSZU1DQKZLCm/57KRbAgRNjmfiXHoFGdmEFw0fdEbPByZZgtCjLfj49pjUPKbLIqKL6Ix2YQKVYWWAP1Ha0aAEa2FcVIqZVfZWZJ5VrAE++TDA3/Am/+R/8Du4AYNa0tC1oYUmXWrP346AQmP/wzPUfiFdaM93k0XoxkXfDZaTHfjti/GUg+zVJnAUdjJHXFlxg7XhucYeYrr+r3jTF7zMvr/tbufKjk79pxf5gVKmNiRog5K3l7TObTcKvrGDjLnbgzfmUzBmAU7uccnD8v+05qpkhxgDEMhUB3BKg+x5SzKu8bCQWB/kLideHZyI6vWBwBKyQGFSEhPjACpRjq628ZO7p1M2TmttcFkL5iQR5uxXhsFMCpDxBarsL3EvqoDjCi4Pe7cavprUK/g8cLyGDj9bAFCojPbktT+IkyMQ2jNHdT3aPrONFaOMK9O8qfC9RBvUrFlL45gFy8/H58CRO0ZBNMyseSSXgO+lPQZjlsXR+htzMenbPGDIacU8Rti+4I2KBxACE/C7cVtKHH1X26P2Qz2rd8CzZHb8+BqIDMDZn1A5KbQIme+kBfdsN9pr2D0Qy2gb2bkF6zwyJqAM31ZDmhE1IM9n3skoH1k5IisP3eGh+uBZWYJWPHRChKhJpgCjJxXtKMhXTGpfAjRBwWFLLp4sWABg4LPPWwJnHL5+oFMKiFN2CtMYATr2A2S9fnRTmAgk3KIRw23g4aKuRHoSk1hZ1OvJH2EBEyQYaBfbgUQOlkiBbSyS9NREJMKQHP1CwqZLzBlStR8KsWCxFpI1Aj7/qn5BMOvKgAWGcw2xPGpPei2DlPTbGY4A9syK2kS04he4IRNbAs4hHYG5Bzj00Gh1TTboIxjUMdxWWqLS1sdJ/saNvfCpl+OGP1CbJiE+RgSjMRSgPJKqJvn90WYaMMKC9NjN4NI4O8sgdPAY3jFV5sOnkfPFdCY/zNTXriTKOGDOKCJCRFdljHBsABLUllJRvP5PqpI5YmGpkAaBCdOUzjsQK2bvwqcqf8DJZKtuv1PJfDS2rmqUFkMqjXUUUjAdGlGd+l0SsYvZoT8MOyU/s5WnMBT2IDuYZbJwFyiEWHCQxfaHD0HhMcDMHea9cCefjW3ZFonKFkD5gNpgkaD7f1CTh7sMd+BEbJisT3acsDIGlDU7MjjH7TGcFsLTDpj0fVccCRhjjg/aidAHxGnTKHliz9/ak4W5768Tba4X7Y8uCqc3K+6AvIK6PpaCy7n+U/2/pqs1U2ZMl8xB0YlJlDbN1nQ6KC+y+9K9phinvcrif5eI4w0ZVvzd7Rex+jiq7jkMJvhquo6Zzkg/YWUGKEPRU3bVL9AFyO5hltYLCgTp2PCEb1GOA8hNn9GVhY69Ocwh9xS9B6vMh2hqlUwMhFwEVG2AoQ0+9Ow840/F/SFJXIqBGYcijJTdVR1yLfOhBUUrSoKTPMwoBCDW/+v0Lkeu1cCVgy2dtPOavncBnDAzacqfB26s48NkKZ1uVNKcJ4IOSN3ZSFMU0Dlhw83uNLw4lCliVEH1o9u553FB2IfOMI4EWbelmrSKFfSROZZsf0QT02atLlBCH4DYqbIaGsebOQ4+YbebeQCxsmcROEbwtk2qwiJgoZPHWMDjA9p5NDx5YT3QGQfuBluIyoLbXZbFU0+XNI2e/0SylFE6O7yKBSnTbAOlcsbbEAoB2Wm5YGYNVEehVrvTG0HX+beAVRHuXPSFnS/lcK13WHLCxqo0ENLqmA4bKjyKdQK30rh/PEVdWhh/F+mMG91QylmXL0kgUIz1U3M/GkKbXVUPFcuBeUn4chmcQoBfUjU+NqGt5kYxuqBd8DRaQ8QkgYI1BBj+unJwf2waAsjdQQUs8CdDh4gtAXw5VCBVoDCnsOIUrl3mAYspuLVBGKMHeBb2DYC8SSrz224v2/5j18htTAgrDbAP0RYsxA0v1uPhVn2katLV5RT6DCi7ig0bSXcLFgDWiOAek7DrPWsNe9fQ20j8mWBokt8LAfiXDFtt8DF79ElZZNDNq18Lk+QOxURUhForCfOhotkzRHAhEqS251YpWkq0wE5SIXYjNj0ranpQ+3GW31uuCS5Nuz21gXmymBSiEB/UI1YKqIVovUM+0qSaUBsBnA+yGabFqb2mkb1jJmxiPA8WIG5JQZqtM62yuGwTZwuUR4/IngNHg+EkgGh1bpdfKfowYMnGRSnHNNBiDC/UihbQk1c6Ic5+CZgeMzJMGep8KsQRO7JCGNqUNNrmuUdmWe85bk6Mx9LfXdaYKrTFBSIRdU0QdC18Y4YrXCUXd+j96kDfDQifCfLZyV6iOdwmasYC2d8tu60FUu5g0ZEDskS30JYeyDOBe0uXSMRJLZyIwBS+x0zCLVm6ZYNHR7+RcGLp8pceUOGY3Pwne0eHUwBJihowhtmbtB5nsxZZyj2bht0Bb2aKQbRiGkosLXNkKsxdIOD+8XcZdzUZ7Y5WioyBxUhGgqs4S1n76ELmu0zj7JRe0tEpjF1dDCw/8tXHGA8BGsPItEJvlYd+/qSWAzdLFD/qLhEozmxAsOkUGfY5W3ksqiz7PLmWE8H6611l/bO2tWmexIoMMMLo9OATpAryIMMWVrTZqX//xI9RmGwHI97u4+R8o4vM08vpgo6H4m+A7Ue48pNKxSXn+dF6MGQ/s8JjA3CBD2t7RaoaLkNZwO7xJ6gy0MNHePpU7b97IYancJzlswY01cMQMEYxsUD/ftPkKtoT6yhJfSSXituQpixRpR3AFbPfmJdoHHpbCkdy7tJjwO50zfM4yuu8r+sQH/kZWhd0CQS5+O4WU7lqBC8+6GLScnZCw2e6E0MGtPhWic0LwXRtOKUpBrIHkbowfvLN2+UMx0YGvKHE2RAKd0DqAJf3jKSDVZ8Fxk4DBbVxJv4QgqBzc6fK7q/S6sxK3oWGVD/im3I9w6oQR3mPDh/ODS1fTGJysGJ0w0UgYjBe4RYRrrJ28fHInoxhdsz5qiFIaZ9mbVnPkBddEvi8Bb9ODipiOzfdA7FuCKsKd9WjF8nzOfU4OAkCnSPM2pOa6D5DQoFjXfCmFUmt7DVXEPqIO8MpTPC4qbgcIwz2qjLdO8hhK05A3cIrU3cOXTDNlEALUZX9ETIZOckHtgOEXbCELY/J1DrO0jMqmgahVxZ3bod8ps7nPtHBG6ii0R9sTxinDxLlSOrj/bJKui7n0MzGMJZfjc8SufcKCbk3DW/vYd1eAKqcVuhOlG4Wwxr66OQ4M1dTCi5WToFIJrAoA6k4PaSZO7TtPVlh1f0ANOEc8Z5ch5fKre7lscVwIcNgmaWI/XrPYmY5pBJfb0cvHcO88Xh463aHSKUFzTVHgZzDE8CEO4Jc2SraBgOeKEXWPaBapjOkRiVfo1to4k3/YJL4tHT0e7ewcubV35G0GS78Mu7CDXDjJd6bfZbiDAIvRrhD21gkPM+r9D325KK8JspJf9VQn1NeWPLB2EOZoV0JUqoo3ghkXRrTx6tQO9SIHukc6DMjTp9zSIXIF/Q3wbOtSNfaYUf/PpAYsELBF4+KqGhIvgGFQwOpLAg/pZgAK+r8PshzbluaBCHBNJvza53vPfvmQBm8wW8kRYVpN2anY1HlJvJWFTIXDTuB8SBcGt2e5XSLrMKuyPIxIpWdSq83tQjeQNBuuTphLiw7N4Qe2lGWN556U4F/QZEYtfNPTJiUSaPEB53v/velGmBRE4pd3M3iHe9eezw+niwkUUv6Uzc+V4sqKVScI7sEwU48+sNZXnd5q3HyAW47PASRoGypLThNy1qnYzDSKXOUrkjMEWHR/1YU2s04JsONJAjgV0ElupvkwetS9s17NSq8huBlkpnMsij1m013vQqwQuB5e7gmUQqo1osOGJX7ieB5YaELhhSr02HLbjQaxgegDInwhF4CdoXkiYQSaWVtVwfOCo9NHvBi3EHCxI8MiOp5KLyE9+D97SUgtqc2N8GhBmJndXRffnVM7AiyhvTvEH0Z8FPKv0iyRx65FuOclUkxIprnpIioyGoM+JhrDyaNzQKU9uI6DJRC8h4PeDRvKE0dLJKcX8XBWpJ14N5Q+j/T0T5V51a0G/SxER6V10UHFFnsvOMHKwNO5qBI77KDlGdE3dIwPbsJ6I/Ip3GZPYpKcLajk8b+A0iJoclKf7HkqvJHNQWkEalpLRC0ThSJM7tUjW8O5bEu6eZaR60R6HVh5rE63Vc2D1kcafk+oAgrGcEGi92F47HmZw/3YjxYGy7gsOBs+7HRJqZHH2bCnSgx4L3Uet+fxKdy9GPCBgA3WZoWuyk+33TYpJ4+zfs3yeGi0pYBEBsFs6brNN49YRITCG87rgK2UjXCJZENpffaaGh0epIYhbnHlyJ1U+LTzsm402lyD2yutf7+LdIFxsm3Y7wXcZl2Twho9XfTt4F2XC3j5UIufT9RJ1aFLhM4AdQG1YXqVRgcfcDbSwRSvLjsv1TpmchvLaqx2YilZ4vwO+FJ2N67sCJNMn2q+XwKQHs70PWaK+Xu+liP+Np5YxYRM35YbXrterf7/T94he/+MUvfvGL/0n8PxO8HWcj0wB/AAAAAElFTkSuQmCC"
LogoComision2="https://www.postdata.gov.co/sites/all/themes/nuboot_radix/logo-crc-blanco.png"
LogoMercadoTIC="https://upload.wikimedia.org/wikipedia/commons/4/41/Noun_project_network_icon_1365244_cc.svg"
# Set page title and favicon.

st.set_page_config(
    page_title="Batería de indicadores telecomunicaciones", page_icon=LogoComision,layout="wide",initial_sidebar_state="expanded")


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

select_mercado = st.sidebar.selectbox('Servicio',
                                    ['Telefonía local','Telefonía móvil', 'Internet fijo','Internet móvil','Televisión por suscripción'])
                              
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

##INTERNET FIJO
   ###ACCESOS
@st.cache(allow_output_mutation=True)
def ReadApiINTFAccesos():
    resourceid = '540ea080-bf16-4d63-911f-3b4814e8e4f1'
    INTF_ACCESOS = pd.DataFrame()
    consulta_anno=['2017','2018','2019','2020','2021','2022']
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
    
   ###INGRESOS 
@st.cache(allow_output_mutation=True)   
def ReadApiINTFIng():
    resourceid = 'd917a68d-9cb9-4257-82f1-74115a4cf629'
    consulta='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid + ''\
             '&filters[anno]=' + consulta_anno + ''\
             '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
             '&group_by=anno,trimestre,id_empresa,empresa'\
             '&sum=ingresos' 
    response_base = urlopen(consulta + '&limit=10000000') 
    json_base = json.loads(response_base.read())
    INTF_ING = pd.DataFrame(json_base['result']['records'])
    INTF_ING.sum_ingresos = INTF_ING.sum_ingresos.astype('int64')
    INTF_ING = INTF_ING.rename(columns={'sum_ingresos':'ingresos'})
    return INTF_ING

##TV POR SUSCRIPCIÓN
   ###INGRESOS
@st.cache(allow_output_mutation=True)      
def ReadApiTVSUSIng():
    resourceid = '1033b0f2-8107-4e04-ae33-8b12882b762d'
    consulta='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid + ''\
             '&filters[anno]=' + consulta_anno + ''\
             '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
             '&group_by=anno,trimestre,id_empresa,empresa'\
             '&sum=ingresos' 
    response_base = urlopen(consulta + '&limit=10000000') 
    json_base = json.loads(response_base.read())
    TVSUS_ING = pd.DataFrame(json_base['result']['records'])
    TVSUS_ING.sum_ingresos = TVSUS_ING.sum_ingresos.astype('int64')
    TVSUS_ING = TVSUS_ING.rename(columns={'sum_ingresos':'ingresos'})
    return TVSUS_ING
   ###SUSCRIPTORES
@st.cache(allow_output_mutation=True)    
def ReadApiTVSUSSus():
    resourceid = '0c4b69a7-734d-432c-9d9b-9dc600d50391'
    consulta='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid + ''\
             '&filters[mes]=3,6,9,12&filters[anno]=' + consulta_anno + ''\
             '&fields[]=anno&fields[]=mes&fields[]=id_operador&fields[]=nombre_operador&fields[]=id_departamento&fields[]=departamento&fields[]=id_municipio&fields[]=municipio'\
             '&group_by=anno,mes,id_operador,nombre_operador,id_departamento,departamento,id_municipio,municipio'\
             '&sum=suscriptores' 
    response_base = urlopen(consulta + '&limit=10000000') 
    json_base = json.loads(response_base.read())
    TV_SUS = pd.DataFrame(json_base['result']['records'])
    TV_SUS.sum_suscriptores = TV_SUS.sum_suscriptores.astype('int64')
    TV_SUS = TV_SUS.rename(columns={'id_operador':'id_empresa','nombre_operador':'empresa','sum_suscriptores':'suscriptores'})
    return TV_SUS  
        
## TELEFONÍA MÓVIL
    #TRÁFICO:
@st.cache(allow_output_mutation=True)    
def ReadApiVOZTraf():
    resourceid = '1384a4d4-42d7-4930-b43c-bf9768c47ccb'
    consulta='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid + ''\
             '&filters[anno]=' + '2017,2018,2019,2020,2021,2022,2023,2024,2025' + ''\
             '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
             '&group_by=anno,trimestre,id_empresa,empresa'\
             '&sum=trafico' 
    response_base = urlopen(consulta + '&limit=10000000') 
    json_base = json.loads(response_base.read())
    VOZ_TRAF = pd.DataFrame(json_base['result']['records'])
    VOZ_TRAF.sum_trafico = VOZ_TRAF.sum_trafico.astype('int64')
    VOZ_TRAF = VOZ_TRAF.rename(columns={'sum_trafico':'trafico'})
    return VOZ_TRAF
    #INGRESOS
@st.cache(allow_output_mutation=True)    
def ReadApiVOZIng():
    consulta_anno:'2017,2018,2019,2020,2021,2022,2023,2024,2025'
    resourceid = '43f0d3a9-cd5c-4f22-a996-74eae6cba9a3'
    consulta='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid + ''\
             '&filters[anno]=' + '2017,2018,2019,2020,2021,2022,2023,2024,2025' + ''\
             '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
             '&group_by=anno,trimestre,id_empresa,empresa'\
             '&sum=ingresos_totales' 
    response_base = urlopen(consulta + '&limit=10000000') 
    json_base = json.loads(response_base.read())
    VOZ_ING = pd.DataFrame(json_base['result']['records'])
    VOZ_ING.sum_ingresos_totales = VOZ_ING.sum_ingresos_totales.astype('int64')
    VOZ_ING = VOZ_ING.rename(columns={'sum_ingresos_totales':'ingresos'})
    return VOZ_ING
    #ABONADOS
@st.cache(allow_output_mutation=True)  
def ReadApiVOZAbo():
    resourceid = '3a9c0304-3795-4c55-a78e-079362373b4d'
    consulta_anno:'2017,2018,2019,2020,2021,2022,2023,2024,2025'
    consulta='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid + ''\
             '&filters[anno]=' + '2017,2018,2019,2020,2021,2022,2023,2024,2025' + ''\
             '&fields[]=anno&fields[]=trimestre&fields[]=id_proveedor&fields[]=proveedor'\
             '&group_by=anno,trimestre,id_proveedor,proveedor'\
             '&sum=abonados' 
    response_base = urlopen(consulta + '&limit=10000000') 
    json_base = json.loads(response_base.read())
    VOZ_ABO = pd.DataFrame(json_base['result']['records'])
    VOZ_ABO.sum_abonados = VOZ_ABO.sum_abonados.astype('int64')
    VOZ_ABO = VOZ_ABO.rename(columns={'id_proveedor':'id_empresa','proveedor':'empresa','sum_abonados':'abonados'})
    return VOZ_ABO

## INTERNET MÓVIL
    #TRAFICO 
@st.cache(allow_output_mutation=True)      
def ReadApiIMTraf():
    resourceid_cf = 'd40c5e75-db56-4ec1-a441-0314c47bd71d'
    consulta_cf='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid_cf + ''\
                '&filters[anno]=' + consulta_anno + ''\
                '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
                '&group_by=anno,trimestre,id_empresa,empresa'\
                '&sum=trafico' 
    response_base_cf = urlopen(consulta_cf + '&limit=10000000') 
    json_base_cf = json.loads(response_base_cf.read())
    IMCF_TRAF = pd.DataFrame(json_base_cf['result']['records'])
    IMCF_TRAF.sum_trafico = IMCF_TRAF.sum_trafico.astype('int64')
    resourceid_dda = 'c0be7034-29f8-4400-be54-c4aafe5df606'
    consulta_dda='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid_dda + ''\
                '&filters[anno]=' + consulta_anno + ''\
                '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
                '&group_by=anno,trimestre,id_empresa,empresa'\
                '&sum=trafico' 
    response_base_dda = urlopen(consulta_dda + '&limit=10000000') 
    json_base_dda = json.loads(response_base_dda.read())
    IMDDA_TRAF = pd.DataFrame(json_base_dda['result']['records'])
    IMDDA_TRAF.sum_trafico = IMDDA_TRAF.sum_trafico.astype('int64')
    IM_TRAF=IMDDA_TRAF.merge(IMCF_TRAF, on=['anno','trimestre','id_empresa','empresa'])
    IM_TRAF['trafico']=IM_TRAF['sum_trafico_y'].fillna(0)+IM_TRAF['sum_trafico_x']
    IM_TRAF.drop(columns=['sum_trafico_y','sum_trafico_x'], inplace=True)
    return IM_TRAF
    #INGRESOS
@st.cache(allow_output_mutation=True) 
def ReadApiIMIng():
    resourceid_cf = '8366e39c-6a14-483a-80f4-7278ceb39f88'
    consulta_cf='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid_cf + ''\
                '&filters[anno]=' + consulta_anno + ''\
                '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
                '&group_by=anno,trimestre,id_empresa,empresa'\
                '&sum=ingresos' 
    response_base_cf = urlopen(consulta_cf + '&limit=10000000') 
    json_base_cf = json.loads(response_base_cf.read())
    IMCF_ING = pd.DataFrame(json_base_cf['result']['records'])
    IMCF_ING.sum_ingresos = IMCF_ING.sum_ingresos.astype('int64')
    resourceid_dda = '60a55889-ba71-45ff-b68f-33b503da36f2'
    consulta_dda='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid_dda + ''\
                '&filters[anno]=' + consulta_anno + ''\
                '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
                '&group_by=anno,trimestre,id_empresa,empresa'\
                '&sum=ingresos' 
    response_base_dda = urlopen(consulta_dda + '&limit=10000000') 
    json_base_dda = json.loads(response_base_dda.read())
    IMDDA_ING = pd.DataFrame(json_base_dda['result']['records'])
    IMDDA_ING.sum_ingresos = IMDDA_ING.sum_ingresos.astype('int64')
    IM_ING=IMDDA_ING.merge(IMCF_ING, on=['anno','trimestre','id_empresa','empresa'])
    IM_ING['ingresos']=IM_ING['sum_ingresos_y'].fillna(0)+IM_ING['sum_ingresos_x']
    IM_ING.drop(columns=['sum_ingresos_y','sum_ingresos_x'], inplace=True)
    return IM_ING
    #ACCESOS
@st.cache(allow_output_mutation=True) 
def ReadApiIMAccesos():
    resourceid_cf = '47d07e20-b257-4aaf-9309-1501c75a826c'
    consulta_cf='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid_cf + ''\
                '&filters[anno]=' + consulta_anno + ''\
                '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
                '&group_by=anno,trimestre,id_empresa,empresa'\
                '&sum=cantidad_suscriptores' 
    response_base_cf = urlopen(consulta_cf + '&limit=10000000') 
    json_base_cf = json.loads(response_base_cf.read())
    IMCF_SUS = pd.DataFrame(json_base_cf['result']['records'])
    IMCF_SUS.sum_cantidad_suscriptores = IMCF_SUS.sum_cantidad_suscriptores.astype('int64')
    resourceid_dda = '3df620f6-deec-42a0-a6af-44ca23c2b73c'
    consulta_dda='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid_dda + ''\
                '&filters[anno]=' + consulta_anno + ''\
                '&fields[]=anno&fields[]=trimestre&fields[]=id_empresa&fields[]=empresa'\
                '&group_by=anno,trimestre,id_empresa,empresa'\
                '&sum=cantidad_abonados' 
    response_base_dda = urlopen(consulta_dda + '&limit=10000000') 
    json_base_dda = json.loads(response_base_dda.read())
    IMDDA_ABO = pd.DataFrame(json_base_dda['result']['records'])
    IMDDA_ABO.sum_cantidad_abonados = IMDDA_ABO.sum_cantidad_abonados.astype('int64')
    IM_ACCESOS=IMDDA_ABO.merge(IMCF_SUS, on=['anno','trimestre','id_empresa','empresa'])
    IM_ACCESOS['accesos']=IM_ACCESOS['sum_cantidad_suscriptores'].fillna(0)+IM_ACCESOS['sum_cantidad_abonados']
    IM_ACCESOS.drop(columns=['sum_cantidad_suscriptores','sum_cantidad_abonados'], inplace=True)
    return IM_ACCESOS
    
@st.cache(allow_output_mutation=True) 
def ReadApiINTFAccesosCorp():
    consulta_anno='2017','2018','2019','2020','2021','2022','2023','2024','2025'
    resourceid = '540ea080-bf16-4d63-911f-3b4814e8e4f1'
    INTF_ACCESOS = pd.DataFrame()
    for anno in consulta_anno:
        consulta='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid + ''\
                 '&filters[id_segmento]=107,108&filters[anno]=' + anno + ''\
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
@st.cache(allow_output_mutation=True)
def ReadApiINTFAccesosRes():
    consulta_anno='2017','2018','2019','2020','2021','2022','2023','2024','2025'
    resourceid = '540ea080-bf16-4d63-911f-3b4814e8e4f1'
    INTF_ACCESOS = pd.DataFrame()
    for anno in consulta_anno:
        consulta='https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resourceid + ''\
                 '&filters[id_segmento]=101,102,103,104,105,106&filters[anno]=' + anno + ''\
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
 
##NUMERO DE HOGARES
Hogares=pd.read_csv("https://raw.githubusercontent.com/sbermudezf/ComisionRegulacionComunicaciones/main/HOGARES.csv",delimiter=';')
Hogares.columns=[x.lower() for x in Hogares.columns]
Hogares.id_municipio=Hogares.id_municipio.astype(str)
Hogares.id_departamento=Hogares.id_departamento.astype(str)
Hogares.anno=Hogares.anno.astype('str')
    
if select_mercado == 'Telefonía local':   
    #st.markdown(r"""<h1 id="logotel"><span class="material-icons material-icons-two-tone" style="font-size:36px">local_phone</span> <span>Telefonía local</span></h1>""",unsafe_allow_html=True) 
    st.title('Telefonía local')
    Trafico=ReadAPITrafTL()
    Ingresos=ReadAPIIngTL()
    Lineas=ReadAPILinTL()
    Trafico.id_municipio=Trafico.id_municipio.str.zfill(5)
    Lineas.id_municipio=Lineas.id_municipio.str.zfill(5)
    Trafico['periodo']=Trafico['anno']+'-T'+Trafico['trimestre']
    Ingresos['periodo']=Ingresos['anno']+'-T'+Ingresos['trimestre']
    Lineas['periodo']=Lineas['anno']+'-T'+Lineas['trimestre']
    Trafnac=Trafico.groupby(['periodo','empresa','id_empresa'])['trafico'].sum().reset_index()
    Ingnac=Ingresos.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
    Linnac=Lineas.groupby(['periodo','empresa','id_empresa'])['lineas'].sum().reset_index()
    PERIODOS=Trafnac['periodo'].unique().tolist()
    
    Trafdpto=Trafico.groupby(['periodo','id_departamento','departamento','empresa','id_empresa'])['trafico'].sum().reset_index()
    Trafdpto=Trafdpto[Trafdpto['trafico']>0]
    Lindpto=Lineas.groupby(['periodo','id_departamento','departamento','empresa','id_empresa'])['lineas'].sum().reset_index()
    Lindpto=Lindpto[Lindpto['lineas']>0]

    
    Trafmuni=Trafico.groupby(['periodo','id_municipio','municipio','departamento','empresa','id_empresa'])['trafico'].sum().reset_index()
    Trafmuni=Trafmuni[Trafmuni['trafico']>0]
    Trafmuni.insert(1,'codigo',Trafmuni['municipio']+' - '+Trafmuni['id_municipio'])
    Trafmuni=Trafmuni.drop(['municipio'],axis=1)
    Linmuni=Lineas.groupby(['periodo','id_municipio','municipio','departamento','empresa','id_empresa'])['lineas'].sum().reset_index()
    Linmuni=Linmuni[Linmuni['lineas']>0]
    Linmuni.insert(1,'codigo',Linmuni['municipio']+' - '+Linmuni['id_municipio'])
    Linmuni=Linmuni.drop(['id_municipio','municipio'],axis=1)
    dfTrafico=[];dfIngresos=[];dfLineas=[]
    dfTrafico2=[];dfIngresos2=[];dfLineas2=[]
    dfTrafico3=[];dfIngresos3=[];dfLineas3=[]
    dfTrafico4=[];dfIngresos4=[];dfLineas4=[]
    
    select_dimension=st.sidebar.selectbox('Ámbito',['Departamental','Municipal','Nacional'])
    
    if select_dimension == 'Nacional':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Penetración','Dominancia'])
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
            st.markdown(" La penetración de mercado mide el grado de utilización o alcance de un producto o servicio en relación con el tamaño del mercado potencial estimado para ese producto o servicio.") 
            with st.expander('Información adicional índice de penetración'):
                st.markdown(r'''El indicador de penetración, de manera general, se puede definir como: ''')
                st.latex(r"""\textrm{Penetracion}(t)=\frac{\textrm{Transacciones}(t)}{\textrm{Tamaño total del mercado}(t)}""")
                st.markdown(r"""En donde las transacciones en el periodo t pueden representarse, en el caso de los mercados de comunicaciones,
            mediante variables como el número de líneas, accesos, conexiones, suscripciones tráfico o envíos.
            Por su parte, el tamaño total del mercado suele ser aproximado mediante variables demográficas como el número de habitantes u hogares, entre otras.""")                    
        if select_indicador == 'Dominancia':
            st.write("### Índice de dominancia")
            st.markdown("El índice de dominancia se calcula de forma similar al IHH, tomando, en lugar de las participaciones directas en el mercado, la participación de cada empresa en el cálculo original del IHH (Lis-Gutiérrez, 2013).")
            with st.expander('Información adicional índice de dominancia'):
                st.write("La fórmula de la dominancia está dada como")
                st.latex(r'''ID=\sum_{i=1}^{n}h_{i}^{2}''')
                st.write(r""" **Donde:**
    -   $h_{i}=S_{i}^{2}/IHH$                 
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    Igual que para el IHH, el rango de valores de éste índice está entre $1/n$ y $1$. Se han establecido rangos de niveles de concentración, asociados con barreras a la entrada, como se muestra en el siguiente cuadro.

    | Concentración                           | Rango          |
    |-----------------------------------------|----------------|
    | Baja barreras a la entrada              | $<0.25$        |
    | Nivel medio de barreras a la entrada    | $0.25 - 0.50$  |
    | Nivel moderado de barreras a la entrada | $0.50 - 0.75$  |
    | Altas barreras a la entrada             | $>0.75$        |                
    """)
                st.markdown("*Fuente: Estos rangos se toman de “Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación”. Martinez, O. J. (2017).*")
    
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
            TrafgroupPart.participacion=TrafgroupPart.participacion.round(2)
            InggroupPart.participacion=InggroupPart.participacion.round(2)
            LingroupPart.participacion=LingroupPart.participacion.round(2)            

            #Gráficas
            fig1=PlotlyStenbacka(TrafgroupPart)
            fig2=PlotlyStenbacka(InggroupPart)
            fig3=PlotlyStenbacka(LingroupPart)
            ##           
            
            if select_variable == "Tráfico":
                AgGrid(TrafgroupPart)
                st.plotly_chart(fig1, use_container_width=True)
            if select_variable == "Ingresos":
                AgGrid(InggroupPart)
                st.plotly_chart(fig2, use_container_width=True)
            if select_variable == "Líneas":
                AgGrid(LingroupPart)
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
            TrafgroupPart3.participacion=TrafgroupPart3.participacion.round(2)
            InggroupPart3.participacion=InggroupPart3.participacion.round(2)
            LingroupPart3.participacion=LingroupPart3.participacion.round(2)            
            IHHTraf=TrafgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            IHHLin=LingroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            
            ##Gráficas
            
            fig7 = PlotlyIHH(IHHTraf)   
            fig8 = PlotlyIHH(IHHIng)
            fig9 = PlotlyIHH(IHHLin)  
            
            if select_variable == "Tráfico":
                AgGrid(TrafgroupPart3)
                st.plotly_chart(fig7,use_container_width=True)
            if select_variable == "Ingresos":
                AgGrid(InggroupPart3)
                st.plotly_chart(fig8,use_container_width=True)
            if select_variable == "Líneas":
                AgGrid(LingroupPart3)
                st.plotly_chart(fig9,use_container_width=True)
                
        if select_indicador == 'Linda':
            dflistTraf2=[];dflistIng2=[];dflistLin2=[]
            
            for elem in PERIODOS:
                dflistTraf2.append(Linda(Trafnac,'trafico',elem))
                dflistIng2.append(Linda(Ingnac,'ingresos',elem))
                dflistLin2.append(Linda(Linnac,'lineas',elem))
            LindTraf=pd.concat(dflistTraf2).reset_index().drop('index',axis=1).fillna(np.nan)
            LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
            LindLin=pd.concat(dflistLin2).reset_index().drop('index',axis=1).fillna(np.nan)     


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

        if select_indicador == 'Penetración':
            HogaresNac=Hogares.groupby(['anno'])['hogares'].sum()  
            LinNac=Lineas.groupby(['periodo'])['lineas'].sum().reset_index()
            LinNac.insert(0,'anno',LinNac.periodo.str.split('-',expand=True)[0])
            PenetracionNac=LinNac.merge(HogaresNac, on=['anno'], how='left')
            PenetracionNac.insert(4,'penetracion',PenetracionNac['lineas']/PenetracionNac['hogares'])
            PenetracionNac.penetracion=PenetracionNac.penetracion.round(3)
            if select_variable=='Líneas':
                fig12=PlotlyPenetracion(PenetracionNac)
                AgGrid(PenetracionNac[['periodo','lineas','hogares','penetracion']])
                st.plotly_chart(fig12,use_container_width=True)
            if select_variable=='Tráfico':
                st.write("El indicador de penetración sólo está definido para la variable de Líneas.")
            if select_variable=='Ingresos':
                st.write("El indicador de penetración sólo está definido para la variable de Líneas.")   

        if select_indicador == 'Dominancia':
            PERIODOS=Trafnac['periodo'].unique().tolist()
            for elem in PERIODOS:
                prTr=Trafnac[Trafnac['periodo']==elem]
                prTr.insert(3,'participacion',(prTr['trafico']/prTr['trafico'].sum())*100)
                prTr.insert(4,'IHH',IHH(prTr,'trafico'))
                prTr.insert(5,'Dominancia',Dominancia(prTr,'trafico'))
                dfTrafico4.append(prTr.sort_values(by='participacion',ascending=False))
                ##
                prIn=Ingnac[Ingnac['periodo']==elem]
                prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                prIn.insert(5,'Dominancia',Dominancia(prIn,'ingresos'))
                dfIngresos4.append(prIn.sort_values(by='participacion',ascending=False))
                ##
                prLi=Linnac[Linnac['periodo']==elem]
                prLi.insert(3,'participacion',(prLi['lineas']/prLi['lineas'].sum())*100)
                prLi.insert(4,'IHH',IHH(prLi,'lineas'))
                prLi.insert(5,'Dominancia',Dominancia(prLi,'lineas'))
                dfLineas4.append(prLi.sort_values(by='participacion',ascending=False))
            TrafgroupPart4=pd.concat(dfTrafico4)
            InggroupPart4=pd.concat(dfIngresos4)
            LingroupPart4=pd.concat(dfLineas4)
            TrafgroupPart4.participacion=TrafgroupPart4.participacion.round(2)
            InggroupPart4.participacion=InggroupPart4.participacion.round(2)
            LingroupPart4.participacion=LingroupPart4.participacion.round(2)
            DomTraf=TrafgroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()
            DomIng=InggroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()
            DomLin=LingroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()
            
            ##Gráficas
            
            fig13 = PlotlyDominancia(DomTraf)   
            fig14 = PlotlyDominancia(DomIng)
            fig15 = PlotlyDominancia(DomLin)  
            
            if select_variable == "Tráfico":
                AgGrid(TrafgroupPart4)
                st.plotly_chart(fig13,use_container_width=True)
            if select_variable == "Ingresos":
                AgGrid(InggroupPart4)
                st.plotly_chart(fig14,use_container_width=True)
            if select_variable == "Líneas":
                AgGrid(LingroupPart4)
                st.plotly_chart(fig15,use_container_width=True)
                            
    if select_dimension == 'Municipal':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Penetración','Dominancia'])
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
            st.markdown(" La penetración de mercado mide el grado de utilización o alcance de un producto o servicio en relación con el tamaño del mercado potencial estimado para ese producto o servicio.") 
            with st.expander('Información adicional índice de penetración'):
                st.markdown(r'''El indicador de penetración, de manera general, se puede definir como: ''')
                st.latex(r"""\textrm{Penetracion}(t)=\frac{\textrm{Transacciones}(t)}{\textrm{Tamaño total del mercado}(t)}""")
                st.markdown(r"""En donde las transacciones en el periodo t pueden representarse, en el caso de los mercados de comunicaciones,
            mediante variables como el número de líneas, accesos, conexiones, suscripciones tráfico o envíos.
            Por su parte, el tamaño total del mercado suele ser aproximado mediante variables demográficas como el número de habitantes u hogares, entre otras.""")                    
        if select_indicador == 'Dominancia':
            st.write("### Índice de dominancia")
            st.markdown("El índice de dominancia se calcula de forma similar al IHH, tomando, en lugar de las participaciones directas en el mercado, la participación de cada empresa en el cálculo original del IHH (Lis-Gutiérrez, 2013).")
            with st.expander('Información adicional índice de dominancia'):
                st.write("La fórmula de la dominancia está dada como")
                st.latex(r'''ID=\sum_{i=1}^{n}h_{i}^{2}''')
                st.write(r""" **Donde:**
    -   $h_{i}=S_{i}^{2}/IHH$                 
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    Igual que para el IHH, el rango de valores de éste índice está entre $1/n$ y $1$. Se han establecido rangos de niveles de concentración, asociados con barreras a la entrada, como se muestra en el siguiente cuadro.

    | Concentración                           | Rango          |
    |-----------------------------------------|----------------|
    | Baja barreras a la entrada              | $<0.25$        |
    | Nivel medio de barreras a la entrada    | $0.25 - 0.50$  |
    | Nivel moderado de barreras a la entrada | $0.50 - 0.75$  |
    | Altas barreras a la entrada             | $>0.75$        |                
    """)
                st.markdown("*Fuente: Estos rangos se toman de “Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación”. Martinez, O. J. (2017).*")
        
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
                AgGrid(TrafgroupPart)
                st.plotly_chart(fig1,use_container_width=True)
                # st.markdown('#### Visualización municipal del Stenbacka')
                # periodoME=st.selectbox('Escoja un periodo para calcular el Stenbacka', PERIODOSTRAF,len(PERIODOSTRAF)-1)
                # dfMap=[];
                # for municipios in MUNICIPIOS:
                    # if Trafmuni[(Trafmuni['codigo']==municipios)&(Trafmuni['periodo']==periodoME)].empty==True:
                        # pass
                    # else:                   
                        # prTr2=Trafmuni[(Trafmuni['codigo']==municipios)&(Trafmuni['periodo']==periodoME)]
                        # prTr2.insert(5,'participacion',Participacion(prTr2,'trafico'))
                        # prTr2.insert(6,'stenbacka',Stenbacka(prTr2,'trafico',gamma))
                        # StenMUNI=prTr2.groupby(['id_municipio','codigo'])['stenbacka'].mean().reset_index()
                        # dfMap.append(StenMUNI) 
                # StenMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                # municipios_df=gdf2.merge(StenMap, on='id_municipio')

                # colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                # tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                # for tile in tiles:
                    # folium.TileLayer(tile).add_to(colombia_map)
                # choropleth=folium.Choropleth(
                    # geo_data=Colombian_MUNI,
                    # data=municipios_df,
                    # columns=['id_municipio', 'stenbacka'],
                    # key_on='feature.properties.MPIO_CCNCT',
                    # fill_color='Reds_r', 
                    # fill_opacity=0.9, 
                    # line_opacity=0.9,
                    # legend_name='Stenbacka',
                    # #bins=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                    # smooth_factor=0).add_to(colombia_map)
                # # Adicionar nombres del departamento
                # style_function = "font-size: 15px; font-weight: bold"
                # choropleth.geojson.add_child(
                    # folium.features.GeoJsonTooltip(['MPIO_CCNCT'], style=style_function, labels=False))
                # folium.LayerControl().add_to(colombia_map)

                # #Adicionar valores 
                # style_function = lambda x: {'fillColor': '#ffffff', 
                                            # 'color':'#000000', 
                                            # 'fillOpacity': 0.1, 
                                            # 'weight': 0.1}
                # highlight_function = lambda x: {'fillColor': '#000000', 
                                                # 'color':'#000000', 
                                                # 'fillOpacity': 0.50, 
                                                # 'weight': 0.1}
                # NIL = folium.features.GeoJson(
                    # data = municipios_df,
                    # style_function=style_function, 
                    # control=False,
                    # highlight_function=highlight_function, 
                    # tooltip=folium.features.GeoJsonTooltip(
                        # fields=['codigo_y','stenbacka'],
                        # aliases=['Nombre-ID Municipio','Stenbacka'],
                        # style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    # )
                # )
                # colombia_map.add_child(NIL)
                # colombia_map.keep_in_front(NIL)
                # col1, col2 ,col3= st.columns([1.5,4,1])
                # with col2:
                    # folium_static(colombia_map,width=480) 
                
                
            if select_variable == "Líneas":
                AgGrid(LingroupPart)
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
                    AgGrid(dTraf)
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
                    col1.AgGrid(dLin)
                    col2.write("**Índice de Linda**")    
                    col2.AgGrid(LindLin)
                    st.plotly_chart(fig11,use_container_width=True)        
                else:
                    lind=st.slider('Seleccionar nivel',2,len(LindconLin),2,1)
                    fig11=PlotlyLinda(LindLin)
                    st.write(LindLin.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconLin[lind-1]]))
                    with st.expander("Mostrar datos"):
                        st.write(dLin)
                    st.plotly_chart(fig11,use_container_width=True)

        if select_indicador == 'Penetración':
            HogaresMuni=Hogares.groupby(['anno','id_municipio'])['hogares'].sum().reset_index()  
            LinMuni=Linmuni[(Linmuni['codigo']==MUNI)]
            LinMuni=LinMuni.groupby(['periodo','codigo'])[['lineas']].sum().reset_index()
            LinMuni.insert(0,'anno',LinMuni.periodo.str.split('-',expand=True)[0])
            LinMuni.insert(2,'id_municipio',LinMuni.codigo.str.split('-',expand=True)[1])
            HogaresMuni.id_municipio=HogaresMuni.id_municipio.astype('int64')
            HogaresMuni.anno=HogaresMuni.anno.astype('int64')
            LinMuni.id_municipio=LinMuni.id_municipio.astype('int64')
            LinMuni.anno=LinMuni.anno.astype('int64')
            PenetracionMuni=LinMuni.merge(HogaresMuni, on=['anno','id_municipio'], how='left')
            PenetracionMuni.insert(6,'penetracion',PenetracionMuni['lineas']/PenetracionMuni['hogares'])
            PenetracionMuni.penetracion=PenetracionMuni.penetracion.round(3)
            if select_variable=='Líneas':
                fig12=PlotlyPenetracion(PenetracionMuni)
                AgGrid(PenetracionMuni[['periodo','codigo','lineas','hogares','penetracion']])
                st.plotly_chart(fig12,use_container_width=True)
            if select_variable=='Tráfico':
                st.write("El indicador de penetración sólo está definido para la variable de Líneas.")
            if select_variable=='Ingresos':
                st.write("El indicador de penetración sólo está definido para la variable de Líneas.")    

        if select_indicador == 'Dominancia':            
            for periodo in PERIODOSTRAF:
                prTr=Trafmuni[(Trafmuni['codigo']==MUNI)&(Trafmuni['periodo']==periodo)]
                prLi=Linmuni[(Linmuni['codigo']==MUNI)&(Linmuni['periodo']==periodo)]
                prTr.insert(3,'participacion',(prTr['trafico']/prTr['trafico'].sum())*100)
                prTr.insert(4,'IHH',IHH(prTr,'trafico'))
                prTr.insert(5,'Dominancia',Dominancia(prTr,'trafico'))
                dfTrafico4.append(prTr.sort_values(by='participacion',ascending=False))
                prLi.insert(3,'participacion',(prLi['lineas']/prLi['lineas'].sum())*100)
                prLi.insert(4,'IHH',IHH(prLi,'lineas'))
                prLi.insert(5,'Dominancia',Dominancia(prLi,'lineas'))
                dfLineas4.append(prLi.sort_values(by='participacion',ascending=False))
            TrafgroupPart4=pd.concat(dfTrafico4)
            LingroupPart4=pd.concat(dfLineas4)
            TrafgroupPart4.participacion=TrafgroupPart4.participacion.round(2)
            LingroupPart4.participacion=LingroupPart4.participacion.round(2)
            
            DomTraf=TrafgroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()
            DomLin=LingroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()    
            
            fig13=PlotlyDominancia(DomTraf)
            fig14=PlotlyDominancia(DomLin)

            if select_variable == "Tráfico":
                st.write(TrafgroupPart4)
                st.plotly_chart(fig13,use_container_width=True)
            if select_variable == "Líneas":
                st.write(LingroupPart4)
                st.plotly_chart(fig14,use_container_width=True)  
                                              
    if select_dimension == 'Departamental':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Media entrópica','Penetración','Dominancia'])
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
        if select_indicador == 'Penetración':
            st.write("### Índice de penetración")
            st.markdown(" La penetración de mercado mide el grado de utilización o alcance de un producto o servicio en relación con el tamaño del mercado potencial estimado para ese producto o servicio.") 
            with st.expander('Información adicional índice de penetración'):
                st.markdown(r'''El indicador de penetración, de manera general, se puede definir como: ''')
                st.latex(r"""\textrm{Penetracion}(t)=\frac{\textrm{Transacciones}(t)}{\textrm{Tamaño total del mercado}(t)}""")
                st.markdown(r"""En donde las transacciones en el periodo t pueden representarse, en el caso de los mercados de comunicaciones,
            mediante variables como el número de líneas, accesos, conexiones, suscripciones tráfico o envíos.
            Por su parte, el tamaño total del mercado suele ser aproximado mediante variables demográficas como el número de habitantes u hogares, entre otras.""")                    
        if select_indicador == 'Dominancia':
            st.write("### Índice de dominancia")
            st.markdown("El índice de dominancia se calcula de forma similar al IHH, tomando, en lugar de las participaciones directas en el mercado, la participación de cada empresa en el cálculo original del IHH (Lis-Gutiérrez, 2013).")
            with st.expander('Información adicional índice de dominancia'):
                st.write("La fórmula de la dominancia está dada como")
                st.latex(r'''ID=\sum_{i=1}^{n}h_{i}^{2}''')
                st.write(r""" **Donde:**
    -   $h_{i}=S_{i}^{2}/IHH$                 
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    Igual que para el IHH, el rango de valores de éste índice está entre $1/n$ y $1$. Se han establecido rangos de niveles de concentración, asociados con barreras a la entrada, como se muestra en el siguiente cuadro.

    | Concentración                           | Rango          |
    |-----------------------------------------|----------------|
    | Baja barreras a la entrada              | $<0.25$        |
    | Nivel medio de barreras a la entrada    | $0.25 - 0.50$  |
    | Nivel moderado de barreras a la entrada | $0.50 - 0.75$  |
    | Altas barreras a la entrada             | $>0.75$        |                
    """)
                st.markdown("*Fuente: Estos rangos se toman de “Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación”. Martinez, O. J. (2017).*")
        
        st.write('#### Agregación departamental') 
        col1, col2 = st.columns(2)
        with col1:
            select_variable = st.selectbox('Variable',['Tráfico','Líneas']) 
            
        DEPARTAMENTOSTRAF=sorted(Trafdpto.departamento.unique().tolist())
        DEPARTAMENTOSLIN=sorted(Lindpto.departamento.unique().tolist())
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
                AgGrid(TrafgroupPart)
                st.plotly_chart(fig1,use_container_width=True)
                st.markdown('#### Visualización departamental del Stenbacka')
                periodoME=st.select_slider('Escoja un periodo para calcular el Stenbacka', PERIODOSTRAF,PERIODOSTRAF[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSTRAF:
                    if Trafdpto[(Trafdpto['departamento']==departamento)&(Trafdpto['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prTr2=Trafdpto[(Trafdpto['departamento']==departamento)&(Trafdpto['periodo']==periodoME)]
                        prTr2.insert(5,'participacion',Participacion(prTr2,'trafico'))
                        prTr2.insert(6,'stenbacka',Stenbacka(prTr2,'trafico',gamma))
                        StenDpto=prTr2.groupby(['id_departamento','departamento'])['stenbacka'].mean().reset_index()
                        dfMap.append(StenDpto) 
                StenMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                
                departamentos_df=gdf.merge(StenMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'stenbacka'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds_r', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Stenbacka',
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
                        fields=['id_departamento','departamento_y','stenbacka'],
                        aliases=['ID Departamento','Departamento','Stenbacka'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480) 
                
            if select_variable == "Líneas":
                AgGrid(LingroupPart)
                st.plotly_chart(fig2,use_container_width=True)     

                st.markdown('#### Visualización departamental del Stenbacka')
                periodoME=st.select_slider('Escoja un periodo para calcular el Stenbacka', PERIODOSLIN,PERIODOSLIN[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSLIN:
                    if Lindpto[(Lindpto['departamento']==departamento)&(Lindpto['periodo']==periodoME)].empty==True:
                        pass
                    else:                    
                        prLi2=Lindpto[(Lindpto['departamento']==departamento)&(Lindpto['periodo']==periodoME)]
                        prLi2.insert(5,'participacion',Participacion(prLi2,'lineas'))
                        prLi2.insert(6,'stenbacka',Stenbacka(prLi2,'lineas',gamma))
                        StenDpto=prLi2.groupby(['id_departamento','departamento'])['stenbacka'].mean().reset_index()
                        dfMap.append(StenDpto) 
                StenMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(StenMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'stenbacka'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds_r', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Stenbacka',
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
                        fields=['id_departamento','departamento_y','stenbacka'],
                        aliases=['ID Departamento','Departamento','Stenbacka'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480) 
        
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
                AgGrid(TrafgroupPart3)
                st.plotly_chart(fig5,use_container_width=True)
                st.markdown('#### Visualización departamental del IHH')
                periodoME=st.select_slider('Escoja un periodo para calcular el IHH', PERIODOSTRAF,PERIODOSTRAF[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSTRAF:
                    if Trafdpto[(Trafdpto['departamento']==departamento)&(Trafdpto['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prTr2=Trafdpto[(Trafdpto['departamento']==departamento)&(Trafdpto['periodo']==periodoME)]
                        prTr2.insert(3,'participacion',Participacion(prTr2,'trafico'))
                        prTr2.insert(4,'IHH',IHH(prTr2,'trafico'))
                        IHHDpto=prTr2.groupby(['id_departamento','departamento'])['IHH'].mean().reset_index()
                        dfMap.append(IHHDpto) 
                IHHMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(IHHMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'IHH'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds_r', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='IHH',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','IHH'],
                        aliases=['ID Departamento','Departamento','IHH'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480) 
              
            if select_variable == "Líneas":
                AgGrid(LingroupPart3)
                st.plotly_chart(fig6,use_container_width=True)    
                st.markdown('#### Visualización departamental del IHH')
                periodoME=st.select_slider('Escoja un periodo para calcular el IHH', PERIODOSLIN,PERIODOSLIN[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSTRAF:
                    if Lindpto[(Lindpto['departamento']==departamento)&(Lindpto['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prLi2=Lindpto[(Lindpto['departamento']==departamento)&(Lindpto['periodo']==periodoME)]
                        prLi2.insert(3,'participacion',Participacion(prLi2,'lineas'))
                        prLi2.insert(4,'IHH',IHH(prLi2,'lineas'))
                        IHHDpto=prLi2.groupby(['id_departamento','departamento'])['IHH'].mean().reset_index()
                        dfMap.append(IHHDpto) 
                IHHMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(IHHMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'IHH'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds_r', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='IHH',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','IHH'],
                        aliases=['ID Departamento','Departamento','IHH'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480)                                 
                            
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
                    AgGrid(dTraf)
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
                st.write(r"""##### <center>Visualización de la evolución de la media entrópica en el departamento seleccionado</center>""",unsafe_allow_html=True)
                st.plotly_chart(fig7,use_container_width=True)
                periodoME=st.select_slider('Escoja un periodo para calcular la media entrópica', PERIODOSTRAF,PERIODOSTRAF[-1])
                MEperiodTableTraf=MediaEntropica(Trafico[(Trafico['departamento']==DPTO)&(Trafico['periodo']==periodoME)],'trafico')[1]                
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
                st.write(r"""##### <center>Visualización de la evolución de la media entrópica en el departamento seleccionado</center>""",unsafe_allow_html=True)
                st.plotly_chart(fig8,use_container_width=True)
                periodoME2=st.select_slider('Escoja un periodo para calcular la media entrópica', PERIODOSLIN,PERIODOSLIN[-1])
                MEperiodTableLin=MediaEntropica(Lineas[(Lineas['departamento']==DPTO)&(Lineas['periodo']==periodoME2)],'lineas')[1] 
                
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
                
                col1, col2 ,= st.columns(2)
                with col1:
                    st.write(r"""###### <center>Visualización de la media entrópica en todos los departamentos y en el periodo seleccionado</center>""",unsafe_allow_html=True)
                    folium_static(colombia_map2,width=480)
                with col2:       
                    st.write(r"""###### <center>Visualización de la participación de los municipios dentro del departamento seleccionado</center>""",unsafe_allow_html=True)
                    st.plotly_chart(fig10,use_container_width=True)                

        if select_indicador == 'Penetración':
            HogaresDpto=Hogares.groupby(['anno','id_departamento'])['hogares'].sum().reset_index()  
            LineasDpto=Lindpto[(Lindpto['departamento']==DPTO)]
            LineasDpto=LineasDpto.groupby(['periodo','id_departamento','departamento'])[['lineas']].sum().reset_index()
            LineasDpto.insert(0,'anno',LineasDpto.periodo.str.split('-',expand=True)[0])
            HogaresDpto.id_departamento=HogaresDpto.id_departamento.astype('int64')
            HogaresDpto.anno=HogaresDpto.anno.astype('int64')
            LineasDpto.id_departamento=LineasDpto.id_departamento.astype('int64')
            LineasDpto.anno=LineasDpto.anno.astype('int64')
            PenetracionDpto=LineasDpto.merge(HogaresDpto, on=['anno','id_departamento'], how='left')
            PenetracionDpto.insert(6,'penetracion',PenetracionDpto['lineas']/PenetracionDpto['hogares'])
            PenetracionDpto.penetracion=PenetracionDpto.penetracion.round(3)
            if select_variable=='Líneas':
                fig12=PlotlyPenetracion(PenetracionDpto)
                AgGrid(PenetracionDpto[['periodo','departamento','lineas','hogares','penetracion']])
                st.plotly_chart(fig12,use_container_width=True)
            if select_variable=='Tráfico':
                st.write("El indicador de penetración sólo está definido para la variable de Líneas.")
            if select_variable=='Ingresos':
                st.write("El indicador de penetración sólo está definido para la variable de Líneas.")  

        if select_indicador == 'Dominancia':
        
            for periodo in PERIODOSTRAF:
                prTr=Trafdpto[(Trafdpto['departamento']==DPTO)&(Trafdpto['periodo']==periodo)]
                prLi=Lindpto[(Lindpto['departamento']==DPTO)&(Lindpto['periodo']==periodo)]
                prTr.insert(3,'participacion',(prTr['trafico']/prTr['trafico'].sum())*100)
                prTr.insert(4,'IHH',IHH(prTr,'trafico'))
                prTr.insert(5,'Dominancia',Dominancia(prTr,'trafico'))
                dfTrafico4.append(prTr.sort_values(by='participacion',ascending=False))
                prLi.insert(3,'participacion',(prLi['lineas']/prLi['lineas'].sum())*100)
                prLi.insert(4,'IHH',IHH(prLi,'lineas'))
                prLi.insert(4,'Dominancia',Dominancia(prLi,'lineas'))
                dfLineas4.append(prLi.sort_values(by='participacion',ascending=False))
            TrafgroupPart4=pd.concat(dfTrafico4)
            LingroupPart4=pd.concat(dfLineas4)
            TrafgroupPart4.participacion=TrafgroupPart4.participacion.round(2)
            LingroupPart4.participacion=LingroupPart4.participacion.round(2)
            DomTraf=TrafgroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()
            DomLin=LingroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()    
            
            fig13=PlotlyDominancia(DomTraf)
            fig14=PlotlyDominancia(DomLin)

            if select_variable == "Tráfico":
                AgGrid(TrafgroupPart4)
                st.plotly_chart(fig13,use_container_width=True)
                st.markdown('#### Visualización departamental de la dominancia')
                periodoME=st.select_slider('Escoja un periodo para calcular la dominancia', PERIODOSTRAF,PERIODOSTRAF[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSTRAF:
                    if Trafdpto[(Trafdpto['departamento']==departamento)&(Trafdpto['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prTr2=Trafdpto[(Trafdpto['departamento']==departamento)&(Trafdpto['periodo']==periodoME)]
                        prTr2.insert(3,'participacion',Participacion(prTr2,'trafico'))
                        prTr2.insert(4,'IHH',IHH(prTr2,'trafico'))
                        prTr2.insert(5,'Dominancia',Dominancia(prTr2,'trafico'))
                        DomDpto=prTr2.groupby(['id_departamento','departamento'])['Dominancia'].mean().reset_index()
                        dfMap.append(DomDpto) 
                DomMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(DomMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'Dominancia'],
                    key_on='feature.properties.DPTO',
                    fill_color='Oranges', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Dominancia',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','Dominancia'],
                        aliases=['ID Departamento','Departamento','Dominancia'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480) 
              
            if select_variable == "Líneas":
                AgGrid(LingroupPart4)
                st.plotly_chart(fig14,use_container_width=True)    
                st.markdown('#### Visualización departamental de la dominancia')
                periodoME=st.select_slider('Escoja un periodo para calcular la dominancia', PERIODOSlIN,PERIODOSLIN[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSTRAF:
                    if Lindpto[(Lindpto['departamento']==departamento)&(Lindpto['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prLi2=Lindpto[(Lindpto['departamento']==departamento)&(Lindpto['periodo']==periodoME)]
                        prLi2.insert(3,'participacion',Participacion(prLi2,'lineas'))
                        prLi2.insert(4,'IHH',IHH(prLi2,'lineas'))
                        prLi2.insert(5,'Dominancia',Dominancia(prLi2,'lineas'))
                        DomDpto=prLi2.groupby(['id_departamento','departamento'])['Dominancia'].mean().reset_index()
                        dfMap.append(DomDpto) 
                DomMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(DomMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'Dominancia'],
                    key_on='feature.properties.DPTO',
                    fill_color='Oranges', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Dominancia',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','Dominancia'],
                        aliases=['ID Departamento','Departamento','Dominancia'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480)                                 
                                         
if select_mercado == "Internet fijo":
    st.title('Internet fijo') 
    AccesosIntCorp=ReadApiINTFAccesosCorp()
    AccesosIntRes=ReadApiINTFAccesosRes()
    IngresosInt=ReadApiINTFIng()
    
    AccesosIntCorp['periodo']=AccesosIntCorp['anno']+'-T'+AccesosIntCorp['trimestre']
    AccesosIntCorp.municipio=AccesosIntCorp.municipio.replace({'MARIQUITA - SAN SEBASTIÁN DE MARIQUITA':'SAN SEBASTIÁN DE MARIQUITA'})
    AccesosIntRes['periodo']=AccesosIntRes['anno']+'-T'+AccesosIntRes['trimestre']
    IngresosInt['periodo']=IngresosInt['anno']+'-T'+IngresosInt['trimestre']

    AccnacIntCorp=AccesosIntCorp.groupby(['periodo','empresa','id_empresa'])['accesos'].sum().reset_index()
    AccnacIntRes=AccesosIntRes.groupby(['periodo','empresa','id_empresa'])['accesos'].sum().reset_index()
    IngnacInt=IngresosInt.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
    PERIODOS=AccnacIntCorp['periodo'].unique().tolist()
    
    AccdptoIntCorp=AccesosIntCorp.groupby(['periodo','id_departamento','departamento','empresa','id_empresa'])['accesos'].sum().reset_index()
    AccdptoIntRes=AccesosIntRes.groupby(['periodo','id_departamento','departamento','empresa','id_empresa'])['accesos'].sum().reset_index()
    AccdptoIntCorp=AccdptoIntCorp[AccdptoIntCorp['accesos']>0]
    AccdptoIntRes=AccdptoIntRes[AccdptoIntRes['accesos']>0]    
 
    AccmuniIntCorp=AccesosIntCorp.groupby(['periodo','id_municipio','municipio','departamento','empresa','id_empresa'])['accesos'].sum().reset_index()
    AccmuniIntCorp=AccmuniIntCorp[AccmuniIntCorp['accesos']>0]
    AccmuniIntCorp.insert(1,'codigo',AccmuniIntCorp['municipio']+' - '+AccmuniIntCorp['id_municipio'])
    AccmuniIntCorp=AccmuniIntCorp.drop(['id_municipio','municipio'],axis=1)
    
    AccmuniIntRes=AccesosIntRes.groupby(['periodo','id_municipio','municipio','departamento','empresa','id_empresa'])['accesos'].sum().reset_index()
    AccmuniIntRes=AccmuniIntRes[AccmuniIntRes['accesos']>0]
    AccmuniIntRes.insert(1,'codigo',AccmuniIntRes['municipio']+' - '+AccmuniIntRes['id_municipio'])
    AccmuniIntRes=AccmuniIntRes.drop(['id_municipio','municipio'],axis=1)
    
    dfAccesosCorp=[];dfAccesosRes=[];dfIngresos=[];
    dfAccesosCorp2=[];dfAccesosRes2=[];dfIngresos2=[];
    dfAccesosCorp3=[];dfAccesosRes3=[];dfIngresos3=[];
    dfAccesosCorp4=[];dfAccesosRes4=[];dfIngresos4=[];
    dfAccesosCorp5=[];dfAccesosRes5=[];dfIngresos5=[];

    select_dimension=st.sidebar.selectbox('Ámbito',['Departamental','Municipal','Nacional'])
    
    if select_dimension == 'Nacional':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Penetración','Dominancia'])
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
            st.markdown(" La penetración de mercado mide el grado de utilización o alcance de un producto o servicio en relación con el tamaño del mercado potencial estimado para ese producto o servicio.") 
            with st.expander('Información adicional índice de penetración'):
                st.markdown(r'''El indicador de penetración, de manera general, se puede definir como: ''')
                st.latex(r"""\textrm{Penetracion}(t)=\frac{\textrm{Transacciones}(t)}{\textrm{Tamaño total del mercado}(t)}""")
                st.markdown(r"""En donde las transacciones en el periodo t pueden representarse, en el caso de los mercados de comunicaciones,
            mediante variables como el número de líneas, accesos, conexiones, suscripciones tráfico o envíos.
            Por su parte, el tamaño total del mercado suele ser aproximado mediante variables demográficas como el número de habitantes u hogares, entre otras.""")                    
        if select_indicador == 'Dominancia':
            st.write("### Índice de dominancia")
            st.markdown("El índice de dominancia se calcula de forma similar al IHH, tomando, en lugar de las participaciones directas en el mercado, la participación de cada empresa en el cálculo original del IHH (Lis-Gutiérrez, 2013).")
            with st.expander('Información adicional índice de dominancia'):
                st.write("La fórmula de la dominancia está dada como")
                st.latex(r'''ID=\sum_{i=1}^{n}h_{i}^{2}''')
                st.write(r""" **Donde:**
    -   $h_{i}=S_{i}^{2}/IHH$                 
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    Igual que para el IHH, el rango de valores de éste índice está entre $1/n$ y $1$. Se han establecido rangos de niveles de concentración, asociados con barreras a la entrada, como se muestra en el siguiente cuadro.

    | Concentración                           | Rango          |
    |-----------------------------------------|----------------|
    | Baja barreras a la entrada              | $<0.25$        |
    | Nivel medio de barreras a la entrada    | $0.25 - 0.50$  |
    | Nivel moderado de barreras a la entrada | $0.50 - 0.75$  |
    | Altas barreras a la entrada             | $>0.75$        |                
    """)
                st.markdown("*Fuente: Estos rangos se toman de “Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación”. Martinez, O. J. (2017).*")

        st.write('#### Agregación nacional') 
        select_variable = st.selectbox('Variable',['Accesos-corporativo','Accesos-residencial','Ingresos']) 

        if select_indicador == 'Stenbacka':
            gamma=st.slider('Seleccionar valor gamma',0.0,1.0,0.1)
            for elem in PERIODOS:
                prAccCorp=AccnacIntCorp[AccnacIntCorp['periodo']==elem]
                prAccCorp.insert(3,'participacion',Participacion(prAccCorp,'accesos'))
                prAccCorp.insert(4,'stenbacka',Stenbacka(prAccCorp,'accesos',gamma))
                dfAccesosCorp.append(prAccCorp.sort_values(by='participacion',ascending=False))
                
                prAccRes=AccnacIntRes[AccnacIntRes['periodo']==elem]
                prAccRes.insert(3,'participacion',Participacion(prAccRes,'accesos'))
                prAccRes.insert(4,'stenbacka',Stenbacka(prAccRes,'accesos',gamma))
                dfAccesosRes.append(prAccRes.sort_values(by='participacion',ascending=False))                
        
                prIn=IngnacInt[IngnacInt['periodo']==elem]
                prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
                prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
                dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))
        
            AccgroupPartCorp=pd.concat(dfAccesosCorp)
            AccgroupPartCorp.participacion=AccgroupPartCorp.participacion.round(4)
            AccgroupPartCorp=AccgroupPartCorp[AccgroupPartCorp['participacion']>0]
            
            AccgroupPartRes=pd.concat(dfAccesosRes)
            AccgroupPartRes.participacion=AccgroupPartRes.participacion.round(4)
            AccgroupPartRes=AccgroupPartRes[AccgroupPartRes['participacion']>0]
            
            InggroupPart=pd.concat(dfIngresos)
            InggroupPart.participacion=InggroupPart.participacion.round(4)
            InggroupPart=InggroupPart[InggroupPart['participacion']>0]

            fig1a=PlotlyStenbacka(AccgroupPartCorp)
            fig1b=PlotlyStenbacka(AccgroupPartRes)
            fig2=PlotlyStenbacka(InggroupPart)          
            
            if select_variable == "Accesos-corporativo":
                AgGrid(AccgroupPartCorp)
                st.plotly_chart(fig1a, use_container_width=True)
            if select_variable == "Accesos-residencial":
                AgGrid(AccgroupPartRes)
                st.plotly_chart(fig1b, use_container_width=True)                                
            if select_variable == "Ingresos":
                AgGrid(InggroupPart)
                st.plotly_chart(fig2, use_container_width=True)

        if select_indicador == 'Concentración':
            dflistAccCorp=[];dflistAccRes=[];dflistIng=[]
            
            for elem in PERIODOS:
                dflistAccCorp.append(Concentracion(AccnacIntCorp,'accesos',elem))
                dflistAccRes.append(Concentracion(AccnacIntRes,'accesos',elem))
                dflistIng.append(Concentracion(IngnacInt,'ingresos',elem))
            ConcAccCorp=pd.concat(dflistAccCorp).fillna(1.0)
            ConcAccRes=pd.concat(dflistAccRes).fillna(1.0)
            ConcIng=pd.concat(dflistIng).fillna(1.0)
     
                        
            if select_variable == "Accesos-corporativo":
                colsconAccCorp=ConcAccCorp.columns.values.tolist()
                conc=st.slider('Seleccionar el número de empresas',1,len(colsconAccCorp)-1,1,1)
                fig4a=PlotlyConcentracion(ConcAccCorp)
                st.write(ConcAccCorp.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconAccCorp[conc]]))
                st.plotly_chart(fig4a,use_container_width=True)
            if select_variable == "Accesos-residencial":
                colsconAccRes=ConcAccRes.columns.values.tolist()
                conc=st.slider('Seleccionar el número de empresas',1,len(colsconAccRes)-1,1,1)
                fig4b=PlotlyConcentracion(ConcAccRes)
                st.write(ConcAccRes.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconAccRes[conc]]))
                st.plotly_chart(fig4b,use_container_width=True)                
            if select_variable == "Ingresos":
                colsconIng=ConcIng.columns.values.tolist()
                conc=st.slider('Seleccione el número de empresas',1,len(colsconIng)-1,1,1)
                fig5=PlotlyConcentracion(ConcIng)
                st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                st.plotly_chart(fig5,use_container_width=True)

        if select_indicador == 'IHH':
            PERIODOS=AccnacIntCorp['periodo'].unique().tolist()
            for elem in PERIODOS:
                prAccCorp=AccnacIntCorp[AccnacIntCorp['periodo']==elem]
                prAccCorp.insert(3,'participacion',(prAccCorp['accesos']/prAccCorp['accesos'].sum())*100)
                prAccCorp.insert(4,'IHH',IHH(prAccCorp,'accesos'))
                dfAccesosCorp3.append(prAccCorp.sort_values(by='participacion',ascending=False))
                ##
                prAccRes=AccnacIntRes[AccnacIntRes['periodo']==elem]
                prAccRes.insert(3,'participacion',(prAccRes['accesos']/prAccRes['accesos'].sum())*100)
                prAccRes.insert(4,'IHH',IHH(prAccRes,'accesos'))
                dfAccesosRes3.append(prAccRes.sort_values(by='participacion',ascending=False))
                ##                
                prIn=IngnacInt[IngnacInt['periodo']==elem]
                prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                dfIngresos3.append(prIn.sort_values(by='participacion',ascending=False))
                ##

            AccgroupPartCorp3=pd.concat(dfAccesosCorp3)
            AccgroupPartRes3=pd.concat(dfAccesosRes3)
            InggroupPart3=pd.concat(dfIngresos3)
            
            IHHAccCorp=AccgroupPartCorp3.groupby(['periodo'])['IHH'].mean().reset_index()
            IHHAccRes=AccgroupPartRes3.groupby(['periodo'])['IHH'].mean().reset_index()
            IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            
            ##Gráficas
            
            fig7a = PlotlyIHH(IHHAccCorp)
            fig7b = PlotlyIHH(IHHAccRes)            
            fig8 = PlotlyIHH(IHHIng)  
            
            if select_variable == "Accesos-corporativo":
                AgGrid(AccgroupPartCorp3)
                st.plotly_chart(fig7a,use_container_width=True)                
            if select_variable == "Accesos-residencial":
                AgGrid(AccgroupPartRes3)
                st.plotly_chart(fig7b,use_container_width=True)                
            if select_variable == "Ingresos":
                AgGrid(InggroupPart3)
                st.plotly_chart(fig8,use_container_width=True)

        if select_indicador == 'Linda':
            dflistAccCorp2=[];dflistAccRes2=[];dflistIng2=[]
            
            for elem in PERIODOS:
                dflistAccCorp2.append(Linda(AccnacIntCorp,'accesos',elem))
                dflistAccRes2.append(Linda(AccnacIntRes,'accesos',elem))
                dflistIng2.append(Linda(IngnacInt,'ingresos',elem))
            LindAccCorp=pd.concat(dflistAccCorp2).reset_index().drop('index',axis=1).fillna(np.nan)
            LindAccRes=pd.concat(dflistAccRes2).reset_index().drop('index',axis=1).fillna(np.nan)
            LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 

            if select_variable == "Accesos-corporativo":
                LindconAccCorp=LindAccCorp.columns.values.tolist()
                lind=st.slider('Seleccionar nivel',2,len(LindconAccCorp),2,1)
                fig10a=PlotlyLinda(LindAccCorp)
                st.write(LindAccCorp.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconAccCorp[lind-1]]))
                st.plotly_chart(fig10a,use_container_width=True)
            if select_variable == "Accesos-residencial":
                LindconAccRes=LindAccRes.columns.values.tolist()
                lind=st.slider('Seleccionar nivel',2,len(LindconAccRes),2,1)
                fig10b=PlotlyLinda(LindAccRes)
                st.write(LindAccRes.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconAccRes[lind-1]]))
                st.plotly_chart(fig10b,use_container_width=True)                
            if select_variable == "Ingresos":
                LindconIng=LindIng.columns.values.tolist()            
                lind=st.slider('Seleccionar nivel',2,len(LindconIng),2,1)
                fig11=PlotlyLinda(LindIng)
                st.write(LindIng.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconIng[lind-1]]))
                st.plotly_chart(fig11,use_container_width=True)

        if select_indicador == 'Penetración':
            HogaresNac=Hogares.groupby(['anno'])['hogares'].sum()  
            AccNac=AccesosIntRes.groupby(['periodo'])['accesos'].sum().reset_index()
            AccNac.insert(0,'anno',AccNac.periodo.str.split('-',expand=True)[0])
            PenetracionNac=AccNac.merge(HogaresNac, on=['anno'], how='left')
            PenetracionNac.insert(4,'penetracion',PenetracionNac['accesos']/PenetracionNac['hogares'])
            PenetracionNac.penetracion=PenetracionNac.penetracion.round(3)
            if select_variable=='Accesos-residencial':
                fig12=PlotlyPenetracion(PenetracionNac)
                AgGrid(PenetracionNac[['periodo','accesos','hogares','penetracion']])
                st.plotly_chart(fig12,use_container_width=True)
            if select_variable=='Accesos-corporativo':
                st.write("El indicador de penetración sólo está definido para la variable de Accesos-residencial.")

        if select_indicador == 'Dominancia':
            PERIODOS=AccnacIntCorp['periodo'].unique().tolist()
            for elem in PERIODOS:
                prAccCorp=AccnacIntCorp[AccnacIntCorp['periodo']==elem]
                prAccCorp.insert(3,'participacion',(prAccCorp['accesos']/prAccCorp['accesos'].sum())*100)
                prAccCorp.insert(4,'IHH',IHH(prAccCorp,'accesos'))
                prAccCorp.insert(5,'Dominancia',Dominancia(prAccCorp,'accesos'))
                dfAccesosCorp4.append(prAccCorp.sort_values(by='participacion',ascending=False))
                ##
                prAccRes=AccnacIntRes[AccnacIntRes['periodo']==elem]
                prAccRes.insert(3,'participacion',(prAccRes['accesos']/prAccRes['accesos'].sum())*100)
                prAccRes.insert(4,'IHH',IHH(prAccRes,'accesos'))
                prAccRes.insert(5,'Dominancia',Dominancia(prAccRes,'accesos'))
                dfAccesosRes4.append(prAccRes.sort_values(by='participacion',ascending=False))
                ##                
                prIn=IngnacInt[IngnacInt['periodo']==elem]
                prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                prIn.insert(5,'Dominancia',Dominancia(prIn,'ingresos'))
                dfIngresos4.append(prIn.sort_values(by='participacion',ascending=False))
                ##

            AccgroupPartCorp4=pd.concat(dfAccesosCorp4)
            AccgroupPartRes4=pd.concat(dfAccesosRes4)
            InggroupPart4=pd.concat(dfIngresos4)
            
            AccgroupPartCorp4.participacion=AccgroupPartCorp4.participacion.round(2)
            AccgroupPartRes4.participacion=AccgroupPartRes4.participacion.round(2)
            InggroupPart4.participacion=InggroupPart4.participacion.round(2)
            
            DomAccCorp=AccgroupPartCorp4.groupby(['periodo'])['Dominancia'].mean().reset_index()
            DomAccRes=AccgroupPartRes4.groupby(['periodo'])['Dominancia'].mean().reset_index()
            DomIng=InggroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()
            
            ##Gráficas
            
            fig13 = PlotlyDominancia(DomAccCorp)
            fig14 = PlotlyDominancia(DomAccRes)            
            fig15 = PlotlyDominancia(DomIng)  
            
            if select_variable == "Accesos-corporativo":
                AgGrid(AccgroupPartCorp4)
                st.plotly_chart(fig13,use_container_width=True)                
            if select_variable == "Accesos-residencial":
                AgGrid(AccgroupPartRes4)
                st.plotly_chart(fig14,use_container_width=True)                
            if select_variable == "Ingresos":
                AgGrid(InggroupPart4)
                st.plotly_chart(fig15,use_container_width=True)
 
    if select_dimension == 'Municipal':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Penetración','Dominancia'])
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
            st.markdown(" La penetración de mercado mide el grado de utilización o alcance de un producto o servicio en relación con el tamaño del mercado potencial estimado para ese producto o servicio.") 
            with st.expander('Información adicional índice de penetración'):
                st.markdown(r'''El indicador de penetración, de manera general, se puede definir como: ''')
                st.latex(r"""\textrm{Penetracion}(t)=\frac{\textrm{Transacciones}(t)}{\textrm{Tamaño total del mercado}(t)}""")
                st.markdown(r"""En donde las transacciones en el periodo t pueden representarse, en el caso de los mercados de comunicaciones,
            mediante variables como el número de líneas, accesos, conexiones, suscripciones tráfico o envíos.
            Por su parte, el tamaño total del mercado suele ser aproximado mediante variables demográficas como el número de habitantes u hogares, entre otras.""")                    
        if select_indicador == 'Dominancia':
            st.write("### Índice de dominancia")
            st.markdown("El índice de dominancia se calcula de forma similar al IHH, tomando, en lugar de las participaciones directas en el mercado, la participación de cada empresa en el cálculo original del IHH (Lis-Gutiérrez, 2013).")
            with st.expander('Información adicional índice de dominancia'):
                st.write("La fórmula de la dominancia está dada como")
                st.latex(r'''ID=\sum_{i=1}^{n}h_{i}^{2}''')
                st.write(r""" **Donde:**
    -   $h_{i}=S_{i}^{2}/IHH$                 
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    Igual que para el IHH, el rango de valores de éste índice está entre $1/n$ y $1$. Se han establecido rangos de niveles de concentración, asociados con barreras a la entrada, como se muestra en el siguiente cuadro.

    | Concentración                           | Rango          |
    |-----------------------------------------|----------------|
    | Baja barreras a la entrada              | $<0.25$        |
    | Nivel medio de barreras a la entrada    | $0.25 - 0.50$  |
    | Nivel moderado de barreras a la entrada | $0.50 - 0.75$  |
    | Altas barreras a la entrada             | $>0.75$        |                
    """)
                st.markdown("*Fuente: Estos rangos se toman de “Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación”. Martinez, O. J. (2017).*")

        st.write('#### Desagregación municipal')
        col1, col2 = st.columns(2)
        with col1:        
            select_variable = st.selectbox('Variable',['Accesos-corporativo','Accesos-residencial'])  
        MUNICIPIOS=sorted(AccmuniIntCorp.codigo.unique().tolist())
        with col2:
            MUNI=st.selectbox('Escoja el municipio', MUNICIPIOS)
        PERIODOSACC=AccmuniIntCorp[AccmuniIntCorp['codigo']==MUNI]['periodo'].unique().tolist()
        
    ## Cálculo de los indicadores 
    
        if select_indicador == 'Stenbacka':                        
            gamma=st.slider('Seleccionar valor gamma',0.0,1.0,0.1)
            
            for periodo in PERIODOSACC:
                prAccCorp=AccmuniIntCorp[(AccmuniIntCorp['codigo']==MUNI)&(AccmuniIntCorp['periodo']==periodo)]
                prAccCorp.insert(5,'participacion',Participacion(prAccCorp,'accesos'))
                prAccCorp.insert(6,'stenbacka',Stenbacka(prAccCorp,'accesos',gamma))
                dfAccesosCorp.append(prAccCorp.sort_values(by='participacion',ascending=False))

                prAccRes=AccmuniIntRes[(AccmuniIntRes['codigo']==MUNI)&(AccmuniIntRes['periodo']==periodo)]
                prAccRes.insert(5,'participacion',Participacion(prAccRes,'accesos'))
                prAccRes.insert(6,'stenbacka',Stenbacka(prAccRes,'accesos',gamma))
                dfAccesosRes.append(prAccRes.sort_values(by='participacion',ascending=False))                
            AccgroupPartCorp=pd.concat(dfAccesosCorp)
            AccgroupPartRes=pd.concat(dfAccesosRes)

            ##Graficas 
            
            fig1a=PlotlyStenbacka(AccgroupPartCorp)
            fig1b=PlotlyStenbacka(AccgroupPartRes)
                  
            if select_variable == "Accesos-corporativo":
                AgGrid(AccgroupPartCorp)
                st.plotly_chart(fig1a,use_container_width=True)
            if select_variable == "Accesos-residencial":
                AgGrid(AccgroupPartRes)
                st.plotly_chart(fig1b,use_container_width=True)                

        if select_indicador == 'Concentración':
            dflistAccCorp=[];dflistAccRes=[];
            
            for periodo in PERIODOSACC:
                prAccCorp=AccmuniIntCorp[(AccmuniIntCorp['codigo']==MUNI)&(AccmuniIntCorp['periodo']==periodo)]
                dflistAccCorp.append(Concentracion(prAccCorp,'accesos',periodo))
                prAccRes=AccmuniIntRes[(AccmuniIntRes['codigo']==MUNI)&(AccmuniIntRes['periodo']==periodo)]
                dflistAccRes.append(Concentracion(prAccRes,'accesos',periodo))                
            ConcAccCorp=pd.concat(dflistAccCorp).fillna(1.0).reset_index().drop('index',axis=1)
            ConcAccRes=pd.concat(dflistAccRes).fillna(1.0).reset_index().drop('index',axis=1)
                        
            if select_variable == "Accesos-corporativo":
                colsconAccCorp=ConcAccCorp.columns.values.tolist()
                value1= len(colsconAccCorp)-1 if len(colsconAccCorp)-1 >1 else 2
                conc=st.slider('Seleccione el número de empresas',1,value1,1,1)
                fig3a = PlotlyConcentracion(ConcAccCorp) 
                st.write(ConcAccCorp.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconAccCorp[conc]]))
                st.plotly_chart(fig3a,use_container_width=True) 
            if select_variable == "Accesos-residencial":
                colsconAccRes=ConcAccRes.columns.values.tolist()
                value1= len(colsconAccRes)-1 if len(colsconAccRes)-1 >1 else 2
                conc=st.slider('Seleccione el número de empresas',1,value1,1,1)
                fig3b = PlotlyConcentracion(ConcAccRes) 
                st.write(ConcAccRes.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconAccRes[conc]]))
                st.plotly_chart(fig3b,use_container_width=True)                 

        if select_indicador == 'IHH':            
            for periodo in PERIODOSACC:
                prAcCorp=AccmuniIntCorp[(AccmuniIntCorp['codigo']==MUNI)&(AccmuniIntCorp['periodo']==periodo)]
                prAcCorp.insert(3,'participacion',(prAcCorp['accesos']/prAcCorp['accesos'].sum())*100)
                prAcCorp.insert(4,'IHH',IHH(prAcCorp,'accesos'))
                dfAccesosCorp3.append(prAcCorp.sort_values(by='participacion',ascending=False))
                prAcRes=AccmuniIntRes[(AccmuniIntRes['codigo']==MUNI)&(AccmuniIntRes['periodo']==periodo)]
                prAcRes.insert(3,'participacion',(prAcRes['accesos']/prAcRes['accesos'].sum())*100)
                prAcRes.insert(4,'IHH',IHH(prAcRes,'accesos'))
                dfAccesosRes3.append(prAcRes.sort_values(by='participacion',ascending=False))                    

            AccgroupPartCorp3=pd.concat(dfAccesosCorp3)
            IHHAccCorp=AccgroupPartCorp3.groupby(['periodo'])['IHH'].mean().reset_index() 
            AccgroupPartRes3=pd.concat(dfAccesosRes3)
            IHHAccRes=AccgroupPartRes3.groupby(['periodo'])['IHH'].mean().reset_index()             
            
            fig5a=PlotlyIHH(IHHAccCorp)
            fig5b=PlotlyIHH(IHHAccRes)

            if select_variable == "Accesos-corporativo":
                AgGrid(AccgroupPartCorp3)
                st.plotly_chart(fig5a,use_container_width=True)
            if select_variable == "Accesos-residencial":
                AgGrid(AccgroupPartRes3)
                st.plotly_chart(fig5b,use_container_width=True)                

        if select_indicador == 'Linda':
            dflistAccCorp2=[];datosAccCorp=[];nempresaAccCorp=[]; dflistAccRes2=[];datosAccRes=[];nempresaAccRes=[];                
            for periodo in PERIODOSACC:
                prAcCorp=AccmuniIntCorp[(AccmuniIntCorp['codigo']==MUNI)&(AccmuniIntCorp['periodo']==periodo)]
                nempresaAccCorp.append(prAcCorp.empresa.nunique())
                dflistAccCorp2.append(Linda(prAcCorp,'accesos',periodo))
                datosAccCorp.append(prAcCorp)    
                prAcRes=AccmuniIntRes[(AccmuniIntRes['codigo']==MUNI)&(AccmuniIntRes['periodo']==periodo)]
                nempresaAccRes.append(prAcRes.empresa.nunique())
                dflistAccRes2.append(Linda(prAcRes,'accesos',periodo))
                datosAccRes.append(prAcRes)                  
                
            NemphisAccCorp=max(nempresaAccCorp)  
            dAccCorp=pd.concat(datosAccCorp).reset_index().drop('index',axis=1)
            LindAccCorp=pd.concat(dflistAccCorp2).reset_index().drop('index',axis=1).fillna(np.nan)
            NemphisAccRes=max(nempresaAccRes)  
            dAccRes=pd.concat(datosAccRes).reset_index().drop('index',axis=1)
            LindAccRes=pd.concat(dflistAccRes2).reset_index().drop('index',axis=1).fillna(np.nan)            
                           
            if select_variable == "Accesos-corporativo":
                LindconAccCorp=LindAccCorp.columns.values.tolist()
                if NemphisAccCorp==1:
                    st.write("El índice de linda no está definido para éste municipio pues cuenta con una sola empresa")
                    st.write(dAccCorp)
                elif  NemphisAccCorp==2:
                    col1, col2 = st.columns([3, 1])
                    fig10a=PlotlyLinda2(LindAccCorp)
                    col1.write("**Datos completos**")                    
                    col1.write(dAccCorp)  
                    col2.write("**Índice de Linda**")
                    col2.write(LindAccCorp)
                    st.plotly_chart(fig10a,use_container_width=True)        
                else:    
                    lind=st.slider('Seleccionar nivel',2,len(LindconAccCorp),2,1)
                    fig10a=PlotlyLinda(LindAccCorp)
                    st.write(LindAccCorp.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconAccCorp[lind-1]]))
                    with st.expander("Mostrar datos"):
                        AgGrid(dAccCorp)                    
                    st.plotly_chart(fig10a,use_container_width=True) 

            if select_variable == "Accesos-residencial":
                LindconAccRes=LindAccRes.columns.values.tolist()
                if NemphisAccRes==1:
                    st.write("El índice de linda no está definido para éste municipio pues cuenta con una sola empresa")
                    st.write(dAccRes)
                elif  NemphisAccRes==2:
                    col1, col2 = st.columns([3, 1])
                    fig10b=PlotlyLinda2(LindAccRes)
                    col1.write("**Datos completos**")                    
                    col1.write(dAccRes)  
                    col2.write("**Índice de Linda**")
                    col2.write(LindAccRes)
                    st.plotly_chart(fig10b,use_container_width=True)        
                else:    
                    lind=st.slider('Seleccionar nivel',2,len(LindconAccRes),2,1)
                    fig10b=PlotlyLinda(LindAccRes)
                    st.write(LindAccRes.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconAccRes[lind-1]]))
                    with st.expander("Mostrar datos"):
                        AgGrid(dAccRes)                    
                    st.plotly_chart(fig10b,use_container_width=True) 

        if select_indicador == 'Penetración':
            HogaresMuni=Hogares.groupby(['anno','id_municipio'])['hogares'].sum().reset_index()  
            AccMuni=AccmuniIntRes[(AccmuniIntRes['codigo']==MUNI)]
            AccMuni=AccMuni.groupby(['periodo','codigo'])[['accesos']].sum().reset_index()
            AccMuni.insert(0,'anno',AccMuni.periodo.str.split('-',expand=True)[0])
            AccMuni.insert(2,'id_municipio',AccMuni.codigo.str.split('-',expand=True)[1])
            HogaresMuni.id_municipio=HogaresMuni.id_municipio.astype('int64')
            HogaresMuni.anno=HogaresMuni.anno.astype('int64')
            AccMuni.id_municipio=AccMuni.id_municipio.astype('int64')
            AccMuni.anno=AccMuni.anno.astype('int64')
            PenetracionMuni=AccMuni.merge(HogaresMuni, on=['anno','id_municipio'], how='left')
            PenetracionMuni.insert(6,'penetracion',PenetracionMuni['accesos']/PenetracionMuni['hogares'])
            PenetracionMuni.penetracion=PenetracionMuni.penetracion.round(3)
            if select_variable=='Accesos-residencial':
                fig12=PlotlyPenetracion(PenetracionMuni)
                AgGrid(PenetracionMuni[['periodo','codigo','accesos','hogares','penetracion']])
                st.plotly_chart(fig12,use_container_width=True)
            if select_variable=='Accesos-corporativo':
                st.write("El indicador de penetración sólo está definido para la variable de Accesos-residencial.")

        if select_indicador == 'Dominancia':            
            for periodo in PERIODOSACC:
                prAcCorp=AccmuniIntCorp[(AccmuniIntCorp['codigo']==MUNI)&(AccmuniIntCorp['periodo']==periodo)]
                prAcCorp.insert(3,'participacion',(prAcCorp['accesos']/prAcCorp['accesos'].sum())*100)
                prAcCorp.insert(4,'IHH',IHH(prAcCorp,'accesos'))
                prAcCorp.insert(5,'Dominancia',Dominancia(prAcCorp,'accesos'))
                dfAccesosCorp4.append(prAcCorp.sort_values(by='participacion',ascending=False))
                prAcRes=AccmuniIntRes[(AccmuniIntRes['codigo']==MUNI)&(AccmuniIntRes['periodo']==periodo)]
                prAcRes.insert(3,'participacion',(prAcRes['accesos']/prAcRes['accesos'].sum())*100)
                prAcRes.insert(4,'IHH',IHH(prAcRes,'accesos'))
                prAcRes.insert(5,'Dominancia',Dominancia(prAcRes,'accesos'))
                dfAccesosRes4.append(prAcRes.sort_values(by='participacion',ascending=False))                    

            AccgroupPartCorp4=pd.concat(dfAccesosCorp4)
            DomAccCorp=AccgroupPartCorp4.groupby(['periodo'])['Dominancia'].mean().reset_index() 
            AccgroupPartRes4=pd.concat(dfAccesosRes4)
            DomAccRes=AccgroupPartRes4.groupby(['periodo'])['Dominancia'].mean().reset_index()             
            
            fig13=PlotlyDominancia(DomAccCorp)
            fig14=PlotlyDominancia(DomAccRes)

            if select_variable == "Accesos-corporativo":
                AgGrid(AccgroupPartCorp4)
                st.plotly_chart(fig13,use_container_width=True)
            if select_variable == "Accesos-residencial":
                AgGrid(AccgroupPartRes4)
                st.plotly_chart(fig14,use_container_width=True)            
                
    if select_dimension == 'Departamental':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Media entrópica','Penetración','Dominancia'])
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
        if select_indicador == 'Penetración':
            st.write("### Índice de penetración")
            st.markdown(" La penetración de mercado mide el grado de utilización o alcance de un producto o servicio en relación con el tamaño del mercado potencial estimado para ese producto o servicio.") 
            with st.expander('Información adicional índice de penetración'):
                st.markdown(r'''El indicador de penetración, de manera general, se puede definir como: ''')
                st.latex(r"""\textrm{Penetracion}(t)=\frac{\textrm{Transacciones}(t)}{\textrm{Tamaño total del mercado}(t)}""")
                st.markdown(r"""En donde las transacciones en el periodo t pueden representarse, en el caso de los mercados de comunicaciones,
            mediante variables como el número de líneas, accesos, conexiones, suscripciones tráfico o envíos.
            Por su parte, el tamaño total del mercado suele ser aproximado mediante variables demográficas como el número de habitantes u hogares, entre otras.""")                    
        if select_indicador == 'Dominancia':
            st.write("### Índice de dominancia")
            st.markdown("El índice de dominancia se calcula de forma similar al IHH, tomando, en lugar de las participaciones directas en el mercado, la participación de cada empresa en el cálculo original del IHH (Lis-Gutiérrez, 2013).")
            with st.expander('Información adicional índice de dominancia'):
                st.write("La fórmula de la dominancia está dada como")
                st.latex(r'''ID=\sum_{i=1}^{n}h_{i}^{2}''')
                st.write(r""" **Donde:**
    -   $h_{i}=S_{i}^{2}/IHH$                 
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    Igual que para el IHH, el rango de valores de éste índice está entre $1/n$ y $1$. Se han establecido rangos de niveles de concentración, asociados con barreras a la entrada, como se muestra en el siguiente cuadro.

    | Concentración                           | Rango          |
    |-----------------------------------------|----------------|
    | Baja barreras a la entrada              | $<0.25$        |
    | Nivel medio de barreras a la entrada    | $0.25 - 0.50$  |
    | Nivel moderado de barreras a la entrada | $0.50 - 0.75$  |
    | Altas barreras a la entrada             | $>0.75$        |                
    """)
                st.markdown("*Fuente: Estos rangos se toman de “Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación”. Martinez, O. J. (2017).*")

        st.write('#### Agregación departamental') 
        col1, col2 = st.columns(2)
        with col1:
            select_variable = st.selectbox('Variable',['Accesos-corporativo','Accesos-residencial']) 
        
        
        DEPARTAMENTOSACC=sorted(AccdptoIntRes.departamento.unique().tolist())
    
        with col2:
            DPTO=st.selectbox('Escoja el departamento', DEPARTAMENTOSACC)
        PERIODOSACC=AccdptoIntCorp[AccdptoIntCorp['departamento']==DPTO]['periodo'].unique().tolist()
        PERIODOSACCRES=AccdptoIntRes[AccdptoIntRes['departamento']==DPTO]['periodo'].unique().tolist()

    ##Cálculo de los indicadores
    
        if select_indicador == 'Stenbacka':
            gamma=st.slider('Seleccionar valor gamma',0.0,1.0,0.1)            
        
            for periodo in PERIODOSACC:
                prAcCorp=AccdptoIntCorp[(AccdptoIntCorp['departamento']==DPTO)&(AccdptoIntCorp['periodo']==periodo)]
                prAcCorp.insert(5,'participacion',Participacion(prAcCorp,'accesos'))
                prAcCorp.insert(6,'stenbacka',Stenbacka(prAcCorp,'accesos',gamma))
                dfAccesosCorp.append(prAcCorp.sort_values(by='participacion',ascending=False))
            AccgroupPartCorp=pd.concat(dfAccesosCorp)    
            
            for periodo in PERIODOSACCRES:            
                prAcRes=AccdptoIntRes[(AccdptoIntRes['departamento']==DPTO)&(AccdptoIntRes['periodo']==periodo)]
                prAcRes.insert(5,'participacion',Participacion(prAcRes,'accesos'))
                prAcRes.insert(6,'stenbacka',Stenbacka(prAcRes,'accesos',gamma))
                dfAccesosRes.append(prAcRes.sort_values(by='participacion',ascending=False))                
            AccgroupPartRes=pd.concat(dfAccesosRes)            

            ##Graficas 
            
            fig1a=PlotlyStenbacka(AccgroupPartCorp)
            fig1b=PlotlyStenbacka(AccgroupPartRes)

            if select_variable == "Accesos-corporativo":
                AgGrid(AccgroupPartCorp)
                st.plotly_chart(fig1a,use_container_width=True)
                st.markdown('#### Visualización departamental del Stenbacka')
                periodoME=st.select_slider('Escoja un periodo para calcular el Stenbacka', PERIODOSACC,PERIODOSACC[-1])
                dfMap=[];                
                for departamento in DEPARTAMENTOSACC:
                    if AccdptoIntCorp[(AccdptoIntCorp['departamento']==departamento)&(AccdptoIntCorp['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prAcCorp=AccdptoIntCorp[(AccdptoIntCorp['departamento']==departamento)&(AccdptoIntCorp['periodo']==periodoME)]
                        prAcCorp.insert(5,'participacion',Participacion(prAcCorp,'accesos'))
                        prAcCorp.insert(6,'stenbacka',Stenbacka(prAcCorp,'accesos',gamma))
                        StenDpto=prAcCorp.groupby(['id_departamento','departamento'])['stenbacka'].mean().reset_index()
                        dfMap.append(StenDpto)                 
                StenMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(StenMap, on='id_departamento')                
                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'stenbacka'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds_r', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Stenbacka',
                    #bins=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
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
                        fields=['id_departamento','departamento_y','stenbacka'],
                        aliases=['ID Departamento','Departamento','Stenbacka'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480) 
                                
            if select_variable == "Accesos-residencial":
                AgGrid(AccgroupPartRes)
                st.plotly_chart(fig1b,use_container_width=True)                
                st.markdown('#### Visualización departamental del Stenbacka')
                periodoME=st.select_slider('Escoja un periodo para calcular el Stenbacka', PERIODOSACCRES,PERIODOSACCRES[-1])
                dfMap=[];                
                for departamento in DEPARTAMENTOSACC:
                    if AccdptoIntRes[(AccdptoIntRes['departamento']==departamento)&(AccdptoIntRes['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prAcRes=AccdptoIntRes[(AccdptoIntRes['departamento']==departamento)&(AccdptoIntRes['periodo']==periodoME)]
                        prAcRes.insert(5,'participacion',Participacion(prAcRes,'accesos'))
                        prAcRes.insert(6,'stenbacka',Stenbacka(prAcRes,'accesos',gamma))
                        StenDpto=prAcRes.groupby(['id_departamento','departamento'])['stenbacka'].mean().reset_index()
                        dfMap.append(StenDpto)                 
                StenMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(StenMap, on='id_departamento')                
                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'stenbacka'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds_r', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Stenbacka',
                    #bins=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
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
                        fields=['id_departamento','departamento_y','stenbacka'],
                        aliases=['ID Departamento','Departamento','Stenbacka'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480) 

        if select_indicador =='Concentración':
            dflistAccCorp=[];dflistAccRes=[];

            for periodo in PERIODOSACC:
                prAcCorp=AccdptoIntCorp[(AccdptoIntCorp['departamento']==DPTO)&(AccdptoIntCorp['periodo']==periodo)]
                dflistAccCorp.append(Concentracion(prAcCorp,'accesos',periodo))
            for periodo in PERIODOSACCRES:    
                prAcRes=AccdptoIntRes[(AccdptoIntRes['departamento']==DPTO)&(AccdptoIntRes['periodo']==periodo)]
                dflistAccRes.append(Concentracion(prAcRes,'accesos',periodo))                
            ConcAccCorp=pd.concat(dflistAccCorp).fillna(1.0).reset_index().drop('index',axis=1)
            ConcAccRes=pd.concat(dflistAccRes).fillna(1.0).reset_index().drop('index',axis=1)
           
            if select_variable == "Accesos-corporativo":
                colsconAccCorp=ConcAccCorp.columns.values.tolist()
                value1= len(colsconAccCorp)-1 if len(colsconAccCorp)-1 >1 else 2 
                conc=st.slider('Seleccionar número de expresas ',1,value1,1,1)
                fig3a = PlotlyConcentracion(ConcAccCorp) 
                st.write(ConcAccCorp.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconAccCorp[conc]]))
                st.plotly_chart(fig3a,use_container_width=True)  
            if select_variable == "Accesos-residencial":
                colsconAccRes=ConcAccRes.columns.values.tolist()
                value1= len(colsconAccRes)-1 if len(colsconAccRes)-1 >1 else 2 
                conc=st.slider('Seleccionar número de expresas ',1,value1,1,1)
                fig3b = PlotlyConcentracion(ConcAccRes) 
                st.write(ConcAccRes.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconAccRes[conc]]))
                st.plotly_chart(fig3b,use_container_width=True)                  

        if select_indicador == 'IHH':
            
            for periodo in PERIODOSACC:
                prAcCorp=AccdptoIntCorp[(AccdptoIntCorp['departamento']==DPTO)&(AccdptoIntCorp['periodo']==periodo)]
                prAcCorp.insert(3,'participacion',(prAcCorp['accesos']/prAcCorp['accesos'].sum())*100)
                prAcCorp.insert(4,'IHH',IHH(prAcCorp,'accesos'))
                dfAccesosCorp3.append(prAcCorp.sort_values(by='participacion',ascending=False))
                prAcRes=AccdptoIntRes[(AccdptoIntRes['departamento']==DPTO)&(AccdptoIntRes['periodo']==periodo)]
                prAcRes.insert(3,'participacion',(prAcRes['accesos']/prAcRes['accesos'].sum())*100)
                prAcRes.insert(4,'IHH',IHH(prAcRes,'accesos'))
                dfAccesosRes3.append(prAcRes.sort_values(by='participacion',ascending=False))
                
            AccgroupPartCorp3=pd.concat(dfAccesosCorp3)
            AccgroupPartRes3=pd.concat(dfAccesosRes3)
            IHHAccCorp=AccgroupPartCorp3.groupby(['periodo'])['IHH'].mean().reset_index()  
            IHHAccRes=AccgroupPartRes3.groupby(['periodo'])['IHH'].mean().reset_index()              
            
            fig5a=PlotlyIHH(IHHAccCorp)
            fig5b=PlotlyIHH(IHHAccRes)

            if select_variable == "Accesos-corporativo":
                AgGrid(AccgroupPartCorp3)
                st.plotly_chart(fig5a,use_container_width=True)
                st.markdown('#### Visualización departamental del IHH')
                periodoME=st.select_slider('Escoja un periodo para calcular el IHH', PERIODOSACC,PERIODOSACC[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSACC:
                    if AccdptoIntCorp[(AccdptoIntCorp['departamento']==departamento)&(AccdptoIntCorp['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prAcCorp=AccdptoIntCorp[(AccdptoIntCorp['departamento']==departamento)&(AccdptoIntCorp['periodo']==periodoME)]
                        prAcCorp.insert(3,'participacion',Participacion(prAcCorp,'accesos'))
                        prAcCorp.insert(4,'IHH',IHH(prAcCorp,'accesos'))
                        IHHDpto=prAcCorp.groupby(['id_departamento','departamento'])['IHH'].mean().reset_index()
                        dfMap.append(IHHDpto) 
                IHHMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(IHHMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'IHH'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='IHH',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','IHH'],
                        aliases=['ID Departamento','Departamento','IHH'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480)
                
            if select_variable == "Accesos-residencial":
                AgGrid(AccgroupPartRes3)
                st.plotly_chart(fig5b,use_container_width=True)    
                st.markdown('#### Visualización departamental del IHH')
                periodoME=st.select_slider('Escoja un periodo para calcular el IHH', PERIODOSACCRES,PERIODOSACCRES[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSACC:
                    if AccdptoIntRes[(AccdptoIntRes['departamento']==departamento)&(AccdptoIntRes['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prAcRes=AccdptoIntRes[(AccdptoIntRes['departamento']==departamento)&(AccdptoIntRes['periodo']==periodoME)]
                        prAcRes.insert(3,'participacion',Participacion(prAcRes,'accesos'))
                        prAcRes.insert(4,'IHH',IHH(prAcRes,'accesos'))
                        IHHDpto=prAcRes.groupby(['id_departamento','departamento'])['IHH'].mean().reset_index()
                        dfMap.append(IHHDpto) 
                IHHMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(IHHMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'IHH'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='IHH',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','IHH'],
                        aliases=['ID Departamento','Departamento','IHH'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480)
                
        if select_indicador == 'Linda':
            dflistAccCorp2=[];datosAccCorp=[];nempresaAccCorp=[]; dflistAccRes2=[];datosAccRes=[];nempresaAccRes=[];       
            for periodo in PERIODOSACC:              
                prAcCorp=AccdptoIntCorp[(AccdptoIntCorp['departamento']==DPTO)&(AccdptoIntCorp['periodo']==periodo)]
                nempresaAccCorp.append(prAcCorp.empresa.nunique())
                dflistAccCorp2.append(Linda(prAcCorp,'accesos',periodo))
                datosAccCorp.append(prAcCorp)
                prAcRes=AccdptoIntRes[(AccdptoIntRes['departamento']==DPTO)&(AccdptoIntRes['periodo']==periodo)]
                nempresaAccRes.append(prAcRes.empresa.nunique())
                dflistAccRes2.append(Linda(prAcRes,'accesos',periodo))
                datosAccRes.append(prAcRes)                

            NemphisAccCorp=max(nempresaAccCorp)
            NemphisAccRes=max(nempresaAccRes)
     
            dAccCorp=pd.concat(datosAccCorp).reset_index().drop('index',axis=1)
            dAccRes=pd.concat(datosAccRes).reset_index().drop('index',axis=1)
            LindAccCorp=pd.concat(dflistAccCorp2).reset_index().drop('index',axis=1).fillna(np.nan)
            LindAccRes=pd.concat(dflistAccRes2).reset_index().drop('index',axis=1).fillna(np.nan)

            if select_variable == "Accesos-corporativo":
                LindconAccCorp=LindAccCorp.columns.values.tolist()
                if NemphisAccCorp==1:
                    st.write("El índice de linda no está definido para éste departamento pues cuenta con una sola empresa")
                    st.write(dAccCorp)
                elif  NemphisAccCorp==2:
                    col1, col2 = st.columns([3, 1])
                    fig10a=PlotlyLinda2(LindAccCorp)
                    col1.write("**Datos completos**")                    
                    col1.write(dAccCorp)  
                    col2.write("**Índice de Linda**")
                    col2.write(LindAccCorp)
                    st.plotly_chart(fig10a,use_container_width=True)        
                else:    
                    lind=st.slider('Seleccionar nivel',2,len(LindconAccCorp),2,1)
                    fig10a=PlotlyLinda(LindAccCorp)
                    st.write(LindAccCorp.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconAccCorp[lind-1]]))
                    with st.expander("Mostrar datos"):
                        st.write(dAccCorp)                    
                    st.plotly_chart(fig10a,use_container_width=True)
                    
            if select_variable == "Accesos-residencial":
                LindconAccRes=LindAccRes.columns.values.tolist()
                if NemphisAccRes==1:
                    st.write("El índice de linda no está definido para éste departamento pues cuenta con una sola empresa")
                    st.write(dAccRes)
                elif  NemphisAccRes==2:
                    col1, col2 = st.columns([3, 1])
                    fig10b=PlotlyLinda2(LindAccRes)
                    col1.write("**Datos completos**")                    
                    col1.write(dAccRes)  
                    col2.write("**Índice de Linda**")
                    col2.write(LindAccRes)
                    st.plotly_chart(fig10b,use_container_width=True)        
                else:    
                    lind=st.slider('Seleccionar nivel',2,len(LindconAccRes),2,1)
                    fig10b=PlotlyLinda(LindAccRes)
                    st.write(LindAccRes.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconAccRes[lind-1]]))
                    with st.expander("Mostrar datos"):
                        st.write(dAccRes)                    
                    st.plotly_chart(fig10b,use_container_width=True)                    

        if select_indicador == 'Media entrópica':

            for periodo in PERIODOSACC:
                prAcCorp=AccesosIntCorp[(AccesosIntCorp['departamento']==DPTO)&(AccesosIntCorp['periodo']==periodo)]
                prAcCorp.insert(4,'media entropica',MediaEntropica(prAcCorp,'accesos')[0])
                dfAccesosCorp.append(prAcCorp)
            for periodo in PERIODOSACCRES:    
                prAcRes=AccesosIntRes[(AccesosIntRes['departamento']==DPTO)&(AccesosIntRes['periodo']==periodo)]
                prAcRes.insert(4,'media entropica',MediaEntropica(prAcRes,'accesos')[0])
                dfAccesosRes.append(prAcRes)  
                
            AccgroupPartCorp=pd.concat(dfAccesosCorp)
            AccgroupPartRes=pd.concat(dfAccesosRes)
            MEDIAENTROPICAACCCORP=AccgroupPartCorp.groupby(['periodo'])['media entropica'].mean().reset_index()    
            MEDIAENTROPICAACCRES=AccgroupPartRes.groupby(['periodo'])['media entropica'].mean().reset_index()  
        
            #Graficas
            
            fig7a=PlotlyMEntropica(MEDIAENTROPICAACCCORP)
            fig7b=PlotlyMEntropica(MEDIAENTROPICAACCRES)
            
            if select_variable == "Accesos-corporativo":
                st.write(r"""##### <center>Visualización de la evolución de la media entrópica en el departamento seleccionado</center>""",unsafe_allow_html=True)
                st.plotly_chart(fig7a,use_container_width=True)
                periodoME=st.select_slider('Escoja un periodo para calcular la media entrópica', PERIODOSACC,PERIODOSACC[-1])
                MEperiodTableAccCorp=MediaEntropica(AccesosIntCorp[(AccesosIntCorp['departamento']==DPTO)&(AccesosIntCorp['periodo']==periodoME)],'accesos')[1]                                 
                dfMapCorp=[];
                for departamento in DEPARTAMENTOSACC:
                    prAcCorp=AccesosIntCorp[(AccesosIntCorp['departamento']==departamento)&(AccesosIntCorp['periodo']==periodoME)]
                    prAcCorp.insert(4,'media entropica',MediaEntropica(prAcCorp,'accesos')[0])
                    prAcCorp2=prAcCorp.groupby(['id_departamento','departamento'])['media entropica'].mean().reset_index()
                    dfMapCorp.append(prAcCorp2)
                AccMapCorp=pd.concat(dfMapCorp).reset_index().drop('index',axis=1)
                colsME=['SIJ','SI','WJ','MED','MEE','MEI','Media entropica'] 
                st.write(MEperiodTableAccCorp.reset_index(drop=True).style.apply(f, axis=0, subset=colsME))
                departamentos_dfCorp=gdf.merge(AccMapCorp, on='id_departamento')
                departamentos_dfCorp['media entropica']=departamentos_dfCorp['media entropica'].round(4)
                colombia_mapCorp = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_mapCorp)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_dfCorp,
                    columns=['id_departamento', 'media entropica'],
                    key_on='feature.properties.DPTO',
                    fill_color='Greens', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Media entrópica',
                    bins=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                    smooth_factor=0).add_to(colombia_mapCorp)
                # Adicionar nombres del departamento
                style_function = "font-size: 15px; font-weight: bold"
                choropleth.geojson.add_child(
                    folium.features.GeoJsonTooltip(['NOMBRE_DPT'], style=style_function, labels=False))
                folium.LayerControl().add_to(colombia_mapCorp)

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
                    data = departamentos_dfCorp,
                    style_function=style_function, 
                    control=False,
                    highlight_function=highlight_function, 
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['id_departamento','departamento_y','media entropica'],
                        aliases=['ID Departamento','Departamento','Media entrópica'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_mapCorp.add_child(NIL)
                colombia_mapCorp.keep_in_front(NIL)
                
                MunicipiosMECorp=MEperiodTableAccCorp.groupby(['municipio'])['WJ'].mean().reset_index()
                MunicipiosMECorp=MunicipiosMECorp[MunicipiosMECorp.WJ!=0]
                MunicipiosMECorp.WJ=MunicipiosMECorp.WJ.round(7)
                              
                fig9a=PlotlyMentropicaTorta(MunicipiosMECorp)
                
                col1, col2= st.columns(2)
                with col1:
                    st.write(r"""###### <center>Visualización de la media entrópica en todos los departamentos y en el periodo seleccionado</center>""",unsafe_allow_html=True)
                    folium_static(colombia_mapCorp,width=480)    
                with col2:
                    st.write(r"""###### <center>Visualización de la participación de los municipios dentro del departamento seleccionado</center>""",unsafe_allow_html=True)                
                    st.plotly_chart(fig9a,use_container_width=True)


            if select_variable == "Accesos-residencial":
                st.write(r"""##### <center>Visualización de la evolución de la media entrópica en el departamento seleccionado</center>""",unsafe_allow_html=True)
                st.plotly_chart(fig7b,use_container_width=True)
                periodoME=st.select_slider('Escoja un periodo para calcular la media entrópica', PERIODOSACCRES,PERIODOSACCRES[-1])
                MEperiodTableAccRes=MediaEntropica(AccesosIntRes[(AccesosIntRes['departamento']==DPTO)&(AccesosIntRes['periodo']==periodoME)],'accesos')[1] 
                                
                dfMapRes=[];
                for departamento in DEPARTAMENTOSACC:
                    if AccesosIntRes[(AccesosIntRes['departamento']==departamento)&(AccesosIntRes['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prAcRes=AccesosIntRes[(AccesosIntRes['departamento']==departamento)&(AccesosIntRes['periodo']==periodoME)]
                        prAcRes.insert(4,'media entropica',MediaEntropica(prAcRes,'accesos')[0])
                        prAcRes2=prAcRes.groupby(['id_departamento','departamento'])['media entropica'].mean().reset_index()
                    dfMapRes.append(prAcRes2)
                AccMapRes=pd.concat(dfMapRes).reset_index().drop('index',axis=1)
                colsME=['SIJ','SI','WJ','MED','MEE','MEI','Media entropica'] 
                st.write(MEperiodTableAccRes.reset_index(drop=True).style.apply(f, axis=0, subset=colsME))
                departamentos_dfRes=gdf.merge(AccMapRes, on='id_departamento')
                departamentos_dfRes['media entropica']=departamentos_dfRes['media entropica'].round(4)
                colombia_mapRes = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_mapRes)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_dfRes,
                    columns=['id_departamento', 'media entropica'],
                    key_on='feature.properties.DPTO',
                    fill_color='Greens', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Media entrópica',
                    bins=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                    smooth_factor=0).add_to(colombia_mapRes)
                # Adicionar nombres del departamento
                style_function = "font-size: 15px; font-weight: bold"
                choropleth.geojson.add_child(
                    folium.features.GeoJsonTooltip(['NOMBRE_DPT'], style=style_function, labels=False))
                folium.LayerControl().add_to(colombia_mapRes)

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
                    data = departamentos_dfRes,
                    style_function=style_function, 
                    control=False,
                    highlight_function=highlight_function, 
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['id_departamento','departamento_y','media entropica'],
                        aliases=['ID Departamento','Departamento','Media entrópica'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_mapRes.add_child(NIL)
                colombia_mapRes.keep_in_front(NIL)
                MunicipiosMERes=MEperiodTableAccRes.groupby(['municipio'])['WJ'].mean().reset_index()
                MunicipiosMERes=MunicipiosMERes[MunicipiosMERes.WJ!=0]
                MunicipiosMERes.WJ=MunicipiosMERes.WJ.round(7)
                
                
                fig9b=PlotlyMentropicaTorta(MunicipiosMERes)
                
                col1, col2= st.columns(2)
                with col1:
                    st.write(r"""###### <center>Visualización de la media entrópica en todos los departamentos y en el periodo seleccionado</center>""",unsafe_allow_html=True)
                    folium_static(colombia_mapRes,width=480)    
                with col2:
                    st.write(r"""###### <center>Visualización de la participación de los municipios dentro del departamento seleccionado</center>""",unsafe_allow_html=True)                
                    st.plotly_chart(fig9b,use_container_width=True)

        if select_indicador == 'Penetración':
            HogaresDpto=Hogares.groupby(['anno','id_departamento'])['hogares'].sum().reset_index()  
            AccDpto=AccdptoIntRes[(AccdptoIntRes['departamento']==DPTO)]
            AccDpto=AccDpto.groupby(['periodo','id_departamento','departamento'])[['accesos']].sum().reset_index()
            AccDpto.insert(0,'anno',AccDpto.periodo.str.split('-',expand=True)[0])
            HogaresDpto.id_departamento=HogaresDpto.id_departamento.astype('int64')
            HogaresDpto.anno=HogaresDpto.anno.astype('int64')
            AccDpto.id_departamento=AccDpto.id_departamento.astype('int64')
            AccDpto.anno=AccDpto.anno.astype('int64')
            PenetracionDpto=AccDpto.merge(HogaresDpto, on=['anno','id_departamento'], how='left')
            PenetracionDpto.insert(6,'penetracion',PenetracionDpto['accesos']/PenetracionDpto['hogares'])
            PenetracionDpto.penetracion=PenetracionDpto.penetracion.round(3)
            if select_variable=='Accesos-residencial':
                fig12=PlotlyPenetracion(PenetracionDpto)
                AgGrid(PenetracionDpto[['periodo','departamento','accesos','hogares','penetracion']])
                st.plotly_chart(fig12,use_container_width=True)
            if select_variable=='Accesos-corporativo':
                st.write("El indicador de penetración sólo está definido para la variable de Accesos-residencial.")
                
        if select_indicador == 'Dominancia':
            
            for periodo in PERIODOSACC:
                prAcCorp=AccdptoIntCorp[(AccdptoIntCorp['departamento']==DPTO)&(AccdptoIntCorp['periodo']==periodo)]
                prAcCorp.insert(3,'participacion',(prAcCorp['accesos']/prAcCorp['accesos'].sum())*100)
                prAcCorp.insert(4,'IHH',IHH(prAcCorp,'accesos'))
                prAcCorp.insert(5,'Dominancia',Dominancia(prAcCorp,'accesos'))
                dfAccesosCorp4.append(prAcCorp.sort_values(by='participacion',ascending=False))
                prAcRes=AccdptoIntRes[(AccdptoIntRes['departamento']==DPTO)&(AccdptoIntRes['periodo']==periodo)]
                prAcRes.insert(3,'participacion',(prAcRes['accesos']/prAcRes['accesos'].sum())*100)
                prAcRes.insert(4,'IHH',IHH(prAcRes,'accesos'))
                prAcRes.insert(5,'Dominancia',Dominancia(prAcRes,'accesos'))
                dfAccesosRes4.append(prAcRes.sort_values(by='participacion',ascending=False))
                
            AccgroupPartCorp4=pd.concat(dfAccesosCorp4)
            AccgroupPartRes4=pd.concat(dfAccesosRes4)
            DomAccCorp=AccgroupPartCorp4.groupby(['periodo'])['Dominancia'].mean().reset_index()  
            DomAccRes=AccgroupPartRes4.groupby(['periodo'])['Dominancia'].mean().reset_index()              
            
            fig13=PlotlyDominancia(DomAccCorp)
            fig14=PlotlyDominancia(DomAccRes)

            if select_variable == "Accesos-corporativo":
                AgGrid(AccgroupPartCorp4)
                st.plotly_chart(fig13,use_container_width=True)
                st.markdown('#### Visualización departamental de la dominancia')
                periodoME=st.select_slider('Escoja un periodo para calcular la dominancia', PERIODOSACC,PERIODOSACC[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSACC:
                    if AccdptoIntCorp[(AccdptoIntCorp['departamento']==departamento)&(AccdptoIntCorp['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prAcCorp=AccdptoIntCorp[(AccdptoIntCorp['departamento']==departamento)&(AccdptoIntCorp['periodo']==periodoME)]
                        prAcCorp.insert(3,'participacion',Participacion(prAcCorp,'accesos'))
                        prAcCorp.insert(4,'IHH',IHH(prAcCorp,'accesos'))
                        prAcCorp.insert(4,'Dominancia',Dominancia(prAcCorp,'accesos'))
                        DomDpto=prAcCorp.groupby(['id_departamento','departamento'])['Dominancia'].mean().reset_index()
                        dfMap.append(DomDpto) 
                DomMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(DomMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'Dominancia'],
                    key_on='feature.properties.DPTO',
                    fill_color='Oranges', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Dominancia',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','Dominancia'],
                        aliases=['ID Departamento','Departamento','Dominancia'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480)
                
            if select_variable == "Accesos-residencial":
                AgGrid(AccgroupPartRes4)
                st.plotly_chart(fig14,use_container_width=True)    
                st.markdown('#### Visualización departamental de la dominancia')
                periodoME=st.select_slider('Escoja un periodo para calcular la dominancia', PERIODOSACCRES,PERIODOSACCRES[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSACC:
                    if AccdptoIntRes[(AccdptoIntRes['departamento']==departamento)&(AccdptoIntRes['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prAcRes=AccdptoIntRes[(AccdptoIntRes['departamento']==departamento)&(AccdptoIntRes['periodo']==periodoME)]
                        prAcRes.insert(3,'participacion',Participacion(prAcRes,'accesos'))
                        prAcRes.insert(4,'IHH',IHH(prAcRes,'accesos'))
                        prAcRes.insert(5,'Dominancia',Dominancia(prAcRes,'accesos'))
                        DomDpto=prAcRes.groupby(['id_departamento','departamento'])['Dominancia'].mean().reset_index()
                        dfMap.append(DomDpto) 
                DomMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(DomMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'Dominancia'],
                    key_on='feature.properties.DPTO',
                    fill_color='Oranges', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Dominancia',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','Dominancia'],
                        aliases=['ID Departamento','Departamento','Dominancia'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480)
                                   
if select_mercado == "Televisión por suscripción":
    st.title("Televisión por suscripción") 
    IngresosTV=ReadApiTVSUSIng() 
    SuscriptoresTV=ReadApiTVSUSSus()
    SuscriptoresTV.departamento.replace({'BOGOTÁ, D.C.':'BOGOTÁ D.C.','CAQUETA':'CAQUETÁ'},inplace=True)
    SuscriptoresTV['trimestre']=(SuscriptoresTV['mes'].astype('int64')-1)//3 +1  
    SuscriptoresTV['periodo']=SuscriptoresTV['anno']+'-T'+SuscriptoresTV['trimestre'].astype('str')
    IngresosTV['periodo']=IngresosTV['anno']+'-T'+IngresosTV['trimestre']

    SusnacTV=SuscriptoresTV.groupby(['periodo','empresa','id_empresa'])['suscriptores'].sum().reset_index()
    IngnacTV=IngresosTV.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
    PERIODOS=SusnacTV['periodo'].unique().tolist()
    
    SusdptoTV=SuscriptoresTV.groupby(['periodo','id_departamento','departamento','empresa','id_empresa'])['suscriptores'].sum().reset_index()
    SusdptoTV=SusdptoTV[SusdptoTV['suscriptores']>0]   
 
    SusmuniTV=SuscriptoresTV.groupby(['periodo','id_municipio','municipio','departamento','empresa','id_empresa'])['suscriptores'].sum().reset_index()
    SusmuniTV=SusmuniTV[SusmuniTV['suscriptores']>0]
    SusmuniTV.insert(1,'codigo',SusmuniTV['municipio']+' - '+SusmuniTV['id_municipio'])
#    SusmuniTV.insert(1,'codigo',SusmuniTV['id_municipio'])
    SusmuniTV=SusmuniTV.drop(['id_municipio','municipio'],axis=1)
    
    dfSuscriptores=[];dfIngresos=[];
    dfSuscriptores2=[];dfIngresos2=[];
    dfSuscriptores3=[];dfIngresos3=[];
    dfSuscriptores4=[];dfIngresos4=[];
    

    select_dimension=st.sidebar.selectbox('Ámbito',['Departamental','Municipal','Nacional'])     
    
    if select_dimension == 'Nacional':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Penetración','Dominancia'])
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
            st.markdown(" La penetración de mercado mide el grado de utilización o alcance de un producto o servicio en relación con el tamaño del mercado potencial estimado para ese producto o servicio.") 
            with st.expander('Información adicional índice de penetración'):
                st.markdown(r'''El indicador de penetración, de manera general, se puede definir como: ''')
                st.latex(r"""\textrm{Penetracion}(t)=\frac{\textrm{Transacciones}(t)}{\textrm{Tamaño total del mercado}(t)}""")
                st.markdown(r"""En donde las transacciones en el periodo t pueden representarse, en el caso de los mercados de comunicaciones,
            mediante variables como el número de líneas, accesos, conexiones, suscripciones tráfico o envíos.
            Por su parte, el tamaño total del mercado suele ser aproximado mediante variables demográficas como el número de habitantes u hogares, entre otras.""")                    
        if select_indicador == 'Dominancia':
            st.write("### Índice de dominancia")
            st.markdown("El índice de dominancia se calcula de forma similar al IHH, tomando, en lugar de las participaciones directas en el mercado, la participación de cada empresa en el cálculo original del IHH (Lis-Gutiérrez, 2013).")
            with st.expander('Información adicional índice de dominancia'):
                st.write("La fórmula de la dominancia está dada como")
                st.latex(r'''ID=\sum_{i=1}^{n}h_{i}^{2}''')
                st.write(r""" **Donde:**
    -   $h_{i}=S_{i}^{2}/IHH$                 
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    Igual que para el IHH, el rango de valores de éste índice está entre $1/n$ y $1$. Se han establecido rangos de niveles de concentración, asociados con barreras a la entrada, como se muestra en el siguiente cuadro.

    | Concentración                           | Rango          |
    |-----------------------------------------|----------------|
    | Baja barreras a la entrada              | $<0.25$        |
    | Nivel medio de barreras a la entrada    | $0.25 - 0.50$  |
    | Nivel moderado de barreras a la entrada | $0.50 - 0.75$  |
    | Altas barreras a la entrada             | $>0.75$        |                
    """)
                st.markdown("*Fuente: Estos rangos se toman de “Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación”. Martinez, O. J. (2017).*")
    
        st.write('#### Agregación nacional') 
        select_variable = st.selectbox('Variable',['Suscriptores','Ingresos']) 

        if select_indicador == 'Stenbacka':
            gamma=st.slider('Seleccionar valor gamma',0.0,1.0,0.1)
            for elem in PERIODOS:
                prSus=SusnacTV[SusnacTV['periodo']==elem]
                prSus.insert(3,'participacion',Participacion(prSus,'suscriptores'))
                prSus.insert(4,'stenbacka',Stenbacka(prSus,'suscriptores',gamma))
                dfSuscriptores.append(prSus.sort_values(by='participacion',ascending=False))
        
                prIn=IngnacTV[IngnacTV['periodo']==elem]
                prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
                prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
                dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))
        
            SusgroupPart=pd.concat(dfSuscriptores)
            SusgroupPart.participacion=SusgroupPart.participacion.round(4)
            SusgroupPart=SusgroupPart[SusgroupPart['participacion']>0]
            InggroupPart=pd.concat(dfIngresos)
            InggroupPart.participacion=InggroupPart.participacion.round(4)
            InggroupPart=InggroupPart[InggroupPart['participacion']>0]

            fig1=PlotlyStenbacka(SusgroupPart)
            fig2=PlotlyStenbacka(InggroupPart)          
            
            if select_variable == "Suscriptores":
                AgGrid(SusgroupPart)
                st.plotly_chart(fig1, use_container_width=True)
            if select_variable == "Ingresos":
                AgGrid(InggroupPart)
                st.plotly_chart(fig2, use_container_width=True)    
                
        if select_indicador == 'Concentración':
            dflistSus=[];dflistIng=[]
            
            for elem in PERIODOS:
                dflistSus.append(Concentracion(SusnacTV,'suscriptores',elem))
                dflistIng.append(Concentracion(IngnacTV,'ingresos',elem))
            ConcSus=pd.concat(dflistSus).fillna(1.0)
            ConcIng=pd.concat(dflistIng).fillna(1.0)
     
                        
            if select_variable == "Suscriptores":
                colsconSus=ConcSus.columns.values.tolist()
                conc=st.slider('Seleccionar el número de empresas',1,len(colsconSus)-1,1,1)
                fig4=PlotlyConcentracion(ConcSus)
                st.write(ConcSus.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconSus[conc]]))
                st.plotly_chart(fig4,use_container_width=True)
            if select_variable == "Ingresos":
                colsconIng=ConcIng.columns.values.tolist()
                conc=st.slider('Seleccione el número de empresas',1,len(colsconIng)-1,1,1)
                fig5=PlotlyConcentracion(ConcIng)
                st.write(ConcIng.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconIng[conc]]))
                st.plotly_chart(fig5,use_container_width=True)
                
        if select_indicador == 'IHH':
            PERIODOS=SusnacTV['periodo'].unique().tolist()
            for elem in PERIODOS:
                prSus=SusnacTV[SusnacTV['periodo']==elem]
                prSus.insert(3,'participacion',(prSus['suscriptores']/prSus['suscriptores'].sum())*100)
                prSus.insert(4,'IHH',IHH(prSus,'suscriptores'))
                dfSuscriptores3.append(prSus.sort_values(by='participacion',ascending=False))
                ##
                prIn=IngnacTV[IngnacTV['periodo']==elem]
                prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                dfIngresos3.append(prIn.sort_values(by='participacion',ascending=False))
                ##

            SusgroupPart3=pd.concat(dfSuscriptores3)
            InggroupPart3=pd.concat(dfIngresos3)
            
            IHHSus=SusgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
            IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()                
            
            ##Gráficas
            
            fig7 = PlotlyIHH(IHHSus)   
            fig8 = PlotlyIHH(IHHIng)  
            
            if select_variable == "Suscriptores":
                AgGrid(SusgroupPart3)
                st.plotly_chart(fig7,use_container_width=True)
            if select_variable == "Ingresos":
                AgGrid(InggroupPart3)
                st.plotly_chart(fig8,use_container_width=True)

        if select_indicador == 'Linda':
            dflistSus2=[];dflistIng2=[]
            
            for elem in PERIODOS:
                dflistSus2.append(Linda(SusnacTV,'suscriptores',elem))
                dflistIng2.append(Linda(IngnacTV,'ingresos',elem))
            LindSus=pd.concat(dflistSus2).reset_index().drop('index',axis=1).fillna(np.nan)
            LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
 
            if select_variable == "Suscriptores":
                LindconSus=LindSus.columns.values.tolist()
                lind=st.slider('Seleccionar nivel',2,len(LindconSus),2,1)
                fig10=PlotlyLinda(LindSus)
                st.write(LindSus.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconSus[lind-1]]))
                st.plotly_chart(fig10,use_container_width=True)
            if select_variable == "Ingresos":
                LindconIng=LindIng.columns.values.tolist()            
                lind=st.slider('Seleccionar nivel',2,len(LindconIng),2,1)
                fig11=PlotlyLinda(LindIng)
                st.write(LindIng.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconIng[lind-1]]))
                st.plotly_chart(fig11,use_container_width=True)

        if select_indicador == 'Penetración':
            HogaresNac=Hogares.groupby(['anno'])['hogares'].sum()  
            SusNac=SuscriptoresTV.groupby(['periodo'])['suscriptores'].sum().reset_index()
            SusNac.insert(0,'anno',SusNac.periodo.str.split('-',expand=True)[0])
            PenetracionNac=SusNac.merge(HogaresNac, on=['anno'], how='left')
            PenetracionNac.insert(4,'penetracion',PenetracionNac['suscriptores']/PenetracionNac['hogares'])
            PenetracionNac.penetracion=PenetracionNac.penetracion.round(3)
            if select_variable=='Suscriptores':
                fig12=PlotlyPenetracion(PenetracionNac)
                AgGrid(PenetracionNac[['periodo','suscriptores','hogares','penetracion']])
                st.plotly_chart(fig12,use_container_width=True)

            if select_variable=='Ingresos':
                st.write("El indicador de penetración sólo está definido para la variable de Líneas.") 

        if select_indicador == 'Dominancia':
            PERIODOS=SusnacTV['periodo'].unique().tolist()
            for elem in PERIODOS:
                prSus=SusnacTV[SusnacTV['periodo']==elem]
                prSus.insert(3,'participacion',(prSus['suscriptores']/prSus['suscriptores'].sum())*100)
                prSus.insert(4,'IHH',IHH(prSus,'suscriptores'))
                prSus.insert(5,'Dominancia',Dominancia(prSus,'suscriptores'))
                dfSuscriptores4.append(prSus.sort_values(by='participacion',ascending=False))
                ##
                prIn=IngnacTV[IngnacTV['periodo']==elem]
                prIn.insert(3,'participacion',(prIn['ingresos']/prIn['ingresos'].sum())*100)
                prIn.insert(4,'IHH',IHH(prIn,'ingresos'))
                prIn.insert(5,'Dominancia',Dominancia(prIn,'ingresos'))
                dfIngresos4.append(prIn.sort_values(by='participacion',ascending=False))
                ##

            SusgroupPart4=pd.concat(dfSuscriptores4)
            SusgroupPart4.participacion=SusgroupPart4.participacion.round(2)
            InggroupPart4=pd.concat(dfIngresos4)
            InggroupPart4.participacion=InggroupPart4.participacion.round(2)
            
            DomSus=SusgroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()
            DomIng=InggroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()                
            
            ##Gráficas
            
            fig13 = PlotlyDominancia(DomSus)   
            fig14 = PlotlyDominancia(DomIng)  
            
            if select_variable == "Suscriptores":
                AgGrid(SusgroupPart4)
                st.plotly_chart(fig13,use_container_width=True)
            if select_variable == "Ingresos":
                AgGrid(InggroupPart4)
                st.plotly_chart(fig14,use_container_width=True)
            
    if select_dimension == 'Municipal':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Penetración','Dominancia'])
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
            st.markdown(" La penetración de mercado mide el grado de utilización o alcance de un producto o servicio en relación con el tamaño del mercado potencial estimado para ese producto o servicio.") 
            with st.expander('Información adicional índice de penetración'):
                st.markdown(r'''El indicador de penetración, de manera general, se puede definir como: ''')
                st.latex(r"""\textrm{Penetracion}(t)=\frac{\textrm{Transacciones}(t)}{\textrm{Tamaño total del mercado}(t)}""")
                st.markdown(r"""En donde las transacciones en el periodo t pueden representarse, en el caso de los mercados de comunicaciones,
            mediante variables como el número de líneas, accesos, conexiones, suscripciones tráfico o envíos.
            Por su parte, el tamaño total del mercado suele ser aproximado mediante variables demográficas como el número de habitantes u hogares, entre otras.""")                    
        if select_indicador == 'Dominancia':
            st.write("### Índice de dominancia")
            st.markdown("El índice de dominancia se calcula de forma similar al IHH, tomando, en lugar de las participaciones directas en el mercado, la participación de cada empresa en el cálculo original del IHH (Lis-Gutiérrez, 2013).")
            with st.expander('Información adicional índice de dominancia'):
                st.write("La fórmula de la dominancia está dada como")
                st.latex(r'''ID=\sum_{i=1}^{n}h_{i}^{2}''')
                st.write(r""" **Donde:**
    -   $h_{i}=S_{i}^{2}/IHH$                 
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    Igual que para el IHH, el rango de valores de éste índice está entre $1/n$ y $1$. Se han establecido rangos de niveles de concentración, asociados con barreras a la entrada, como se muestra en el siguiente cuadro.

    | Concentración                           | Rango          |
    |-----------------------------------------|----------------|
    | Baja barreras a la entrada              | $<0.25$        |
    | Nivel medio de barreras a la entrada    | $0.25 - 0.50$  |
    | Nivel moderado de barreras a la entrada | $0.50 - 0.75$  |
    | Altas barreras a la entrada             | $>0.75$        |                
    """)
                st.markdown("*Fuente: Estos rangos se toman de “Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación”. Martinez, O. J. (2017).*")

        st.write('#### Desagregación municipal')
        col1, col2 = st.columns(2)
        with col1:        
            select_variable = st.selectbox('Variable',['Suscriptores'])  
        MUNICIPIOS=sorted(SusmuniTV.codigo.unique().tolist())
        with col2:
            MUNI=st.selectbox('Escoja el municipio', MUNICIPIOS)
        PERIODOSSUS=SusmuniTV[SusmuniTV['codigo']==MUNI]['periodo'].unique().tolist()
        
    ## Cálculo de los indicadores 

        if select_indicador == 'Stenbacka':                        
            gamma=st.slider('Seleccionar valor gamma',0.0,1.0,0.1)
            for periodo in PERIODOSSUS:
                prSus=SusmuniTV[(SusmuniTV['codigo']==MUNI)&(SusmuniTV['periodo']==periodo)]
                prSus.insert(5,'participacion',Participacion(prSus,'suscriptores'))
                prSus.insert(6,'stenbacka',Stenbacka(prSus,'suscriptores',gamma))
                dfSuscriptores.append(prSus.sort_values(by='participacion',ascending=False))
            SusgroupPart=pd.concat(dfSuscriptores)

            ##Graficas 
            
            fig1=PlotlyStenbacka(SusgroupPart)
                  
            if select_variable == "Suscriptores":
                AgGrid(SusgroupPart)
                st.plotly_chart(fig1,use_container_width=True)

        if select_indicador == 'Concentración':
            dflistSus=[]
            
            for periodo in PERIODOSSUS:
                prSus=SusmuniTV[(SusmuniTV['codigo']==MUNI)&(SusmuniTV['periodo']==periodo)]
                dflistSus.append(Concentracion(prSus,'suscriptores',periodo))
            ConcSus=pd.concat(dflistSus).fillna(1.0).reset_index().drop('index',axis=1)
                        
            if select_variable == "Suscriptores":
                colsconSus=ConcSus.columns.values.tolist()
                value1= len(colsconSus)-1 if len(colsconSus)-1 >1 else 2
                conc=st.slider('Seleccione el número de empresas',1,value1,1,1)
                fig3 = PlotlyConcentracion(ConcSus) 
                st.write(ConcSus.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconSus[conc]]))
                st.plotly_chart(fig3,use_container_width=True)  
               
        if select_indicador == 'IHH':            
            for periodo in PERIODOSSUS:
                prSus=SusmuniTV[(SusmuniTV['codigo']==MUNI)&(SusmuniTV['periodo']==periodo)]
                prSus.insert(3,'participacion',(prSus['suscriptores']/prSus['suscriptores'].sum())*100)
                prSus.insert(4,'IHH',IHH(prSus,'suscriptores'))
                dfSuscriptores3.append(prSus.sort_values(by='participacion',ascending=False))

            SusgroupPart3=pd.concat(dfSuscriptores3)
            IHHSus=SusgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()  
            
            fig5=PlotlyIHH(IHHSus)

            if select_variable == "Suscriptores":
                AgGrid(SusgroupPart3)
                st.plotly_chart(fig5,use_container_width=True)               
                
        if select_indicador == 'Linda':
            dflistSus2=[];datosSus=[];nempresaSus=[];                
            for periodo in PERIODOSSUS:
                prSus=SusmuniTV[(SusmuniTV['codigo']==MUNI)&(SusmuniTV['periodo']==periodo)]
                nempresaSus.append(prSus.empresa.nunique())
                dflistSus2.append(Linda(prSus,'suscriptores',periodo))
                datosSus.append(prSus)    
            NemphisSus=max(nempresaSus)  
            dSus=pd.concat(datosSus).reset_index().drop('index',axis=1)
            LindSus=pd.concat(dflistSus2).reset_index().drop('index',axis=1).fillna(np.nan)
                           
            if select_variable == "Suscriptores":
                LindconSus=LindSus.columns.values.tolist()
                if NemphisSus==1:
                    st.write("El índice de linda no está definido para éste municipio pues cuenta con una sola empresa")
                    st.write(dSus)
                elif  NemphisSus==2:
                    col1, col2 = st.columns([3, 1])
                    fig10=PlotlyLinda2(LindSus)
                    col1.write("**Datos completos**")                    
                    col1.write(dSus)  
                    col2.write("**Índice de Linda**")
                    col2.write(LindSus)
                    st.plotly_chart(fig10,use_container_width=True)        
                else:    
                    lind=st.slider('Seleccionar nivel',2,len(LindconSus),2,1)
                    fig10=PlotlyLinda(LindSus)
                    st.write(LindSus.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconSus[lind-1]]))
                    with st.expander("Mostrar datos"):
                        AgGrid(dSus)                    
                    st.plotly_chart(fig10,use_container_width=True) 

        if select_indicador == 'Penetración':
            HogaresMuni=Hogares.groupby(['anno','id_municipio'])['hogares'].sum().reset_index()  
            SusMuni=SusmuniTV[(SusmuniTV['codigo']==MUNI)]
            SusMuni=SusMuni.groupby(['periodo','codigo'])[['suscriptores']].sum().reset_index()
            SusMuni.insert(0,'anno',SusMuni.periodo.str.split('-',expand=True)[0])
            SusMuni.insert(2,'id_municipio',SusMuni.codigo.str.split('-',expand=True)[1])
            HogaresMuni.id_municipio=HogaresMuni.id_municipio.astype('int64')
            HogaresMuni.anno=HogaresMuni.anno.astype('int64')
            SusMuni.id_municipio=SusMuni.id_municipio.astype('int64')
            SusMuni.anno=SusMuni.anno.astype('int64')
            PenetracionMuni=SusMuni.merge(HogaresMuni, on=['anno','id_municipio'], how='left')
            PenetracionMuni.insert(6,'penetracion',PenetracionMuni['suscriptores']/PenetracionMuni['hogares'])
            PenetracionMuni.penetracion=PenetracionMuni.penetracion.round(3)
            if select_variable=='Suscriptores':
                fig12=PlotlyPenetracion(PenetracionMuni)
                AgGrid(PenetracionMuni[['periodo','codigo','suscriptores','hogares','penetracion']])
                st.plotly_chart(fig12,use_container_width=True)

        if select_indicador == 'Dominancia':            
            for periodo in PERIODOSSUS:
                prSus=SusmuniTV[(SusmuniTV['codigo']==MUNI)&(SusmuniTV['periodo']==periodo)]
                prSus.insert(3,'participacion',(prSus['suscriptores']/prSus['suscriptores'].sum())*100)
                prSus.insert(4,'IHH',IHH(prSus,'suscriptores'))
                prSus.insert(5,'Dominancia',Dominancia(prSus,'suscriptores'))
                dfSuscriptores4.append(prSus.sort_values(by='participacion',ascending=False))

            SusgroupPart4=pd.concat(dfSuscriptores4)
            SusgroupPart4.participacion=SusgroupPart4.participacion.round(2)
            DomSus=SusgroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()  
            
            fig13=PlotlyDominancia(DomSus)

            if select_variable == "Suscriptores":
                AgGrid(SusgroupPart4)
                st.plotly_chart(fig13,use_container_width=True)               
                                    
    if select_dimension == 'Departamental':
        select_indicador = st.sidebar.selectbox('Indicador',
                                    ['Stenbacka', 'Concentración','IHH','Linda','Media entrópica','Penetración','Dominancia'])
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
        if select_indicador == 'Penetración':
            st.write("### Índice de penetración")
            st.markdown(" La penetración de mercado mide el grado de utilización o alcance de un producto o servicio en relación con el tamaño del mercado potencial estimado para ese producto o servicio.") 
            with st.expander('Información adicional índice de penetración'):
                st.markdown(r'''El indicador de penetración, de manera general, se puede definir como: ''')
                st.latex(r"""\textrm{Penetracion}(t)=\frac{\textrm{Transacciones}(t)}{\textrm{Tamaño total del mercado}(t)}""")
                st.markdown(r"""En donde las transacciones en el periodo t pueden representarse, en el caso de los mercados de comunicaciones,
            mediante variables como el número de líneas, accesos, conexiones, suscripciones tráfico o envíos.
            Por su parte, el tamaño total del mercado suele ser aproximado mediante variables demográficas como el número de habitantes u hogares, entre otras.""")                    
        if select_indicador == 'Dominancia':
            st.write("### Índice de dominancia")
            st.markdown("El índice de dominancia se calcula de forma similar al IHH, tomando, en lugar de las participaciones directas en el mercado, la participación de cada empresa en el cálculo original del IHH (Lis-Gutiérrez, 2013).")
            with st.expander('Información adicional índice de dominancia'):
                st.write("La fórmula de la dominancia está dada como")
                st.latex(r'''ID=\sum_{i=1}^{n}h_{i}^{2}''')
                st.write(r""" **Donde:**
    -   $h_{i}=S_{i}^{2}/IHH$                 
    -   $S_{i}$ es la participación de mercado de la variable analizada.
    -   $n$ es el número de empresas más grandes consideradas.

    Igual que para el IHH, el rango de valores de éste índice está entre $1/n$ y $1$. Se han establecido rangos de niveles de concentración, asociados con barreras a la entrada, como se muestra en el siguiente cuadro.

    | Concentración                           | Rango          |
    |-----------------------------------------|----------------|
    | Baja barreras a la entrada              | $<0.25$        |
    | Nivel medio de barreras a la entrada    | $0.25 - 0.50$  |
    | Nivel moderado de barreras a la entrada | $0.50 - 0.75$  |
    | Altas barreras a la entrada             | $>0.75$        |                
    """)
                st.markdown("*Fuente: Estos rangos se toman de “Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación”. Martinez, O. J. (2017).*")
                                                
        st.write('#### Agregación departamental') 
        col1, col2 = st.columns(2)
        with col1:
            select_variable = st.selectbox('Variable',['Suscriptores']) 

        DEPARTAMENTOSSUS=sorted(SusdptoTV.departamento.unique().tolist())
        DEPARTAMENTOSSUS.remove('NA')
    
        with col2:
            DPTO=st.selectbox('Escoja el departamento', DEPARTAMENTOSSUS,5)   
        PERIODOSSUS=SusdptoTV[SusdptoTV['departamento']==DPTO]['periodo'].unique().tolist()
        
    ##Cálculo de los indicadores

        if select_indicador == 'Stenbacka':
            gamma=st.slider('Seleccionar valor gamma',0.0,1.0,0.1)            
        
            for periodo in PERIODOSSUS:
                prSus=SusdptoTV[(SusdptoTV['departamento']==DPTO)&(SusdptoTV['periodo']==periodo)]
                prSus.insert(5,'participacion',Participacion(prSus,'suscriptores'))
                prSus.insert(6,'stenbacka',Stenbacka(prSus,'suscriptores',gamma))
                dfSuscriptores.append(prSus.sort_values(by='participacion',ascending=False))
            SusgroupPart=pd.concat(dfSuscriptores) 

            ##Graficas 
            
            fig1=PlotlyStenbacka(SusgroupPart)

            if select_variable == "Suscriptores":
                AgGrid(SusgroupPart)
                st.plotly_chart(fig1,use_container_width=True)
                st.markdown('#### Visualización departamental del Stenbacka')
                periodoME=st.select_slider('Escoja un periodo para calcular el Stenbacka', PERIODOSSUS,PERIODOSSUS[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSSUS:
                    if SusdptoTV[(SusdptoTV['departamento']==departamento)&(SusdptoTV['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prSus2=SusdptoTV[(SusdptoTV['departamento']==departamento)&(SusdptoTV['periodo']==periodoME)]
                        prSus2.insert(5,'participacion',Participacion(prSus2,'suscriptores'))
                        prSus2.insert(6,'stenbacka',Stenbacka(prSus2,'suscriptores',gamma))
                        StenDpto=prSus2.groupby(['id_departamento','departamento'])['stenbacka'].mean().reset_index()
                        dfMap.append(StenDpto) 
                StenMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                
                departamentos_df=gdf.merge(StenMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'stenbacka'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds_r', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Stenbacka',
                    #bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
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
                        fields=['id_departamento','departamento_y','stenbacka'],
                        aliases=['ID Departamento','Departamento','Stenbacka'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480)                
                
        if select_indicador =='Concentración':
            dflistSus=[];

            for periodo in PERIODOSSUS:
                prSus=SusdptoTV[(SusdptoTV['departamento']==DPTO)&(SusdptoTV['periodo']==periodo)]
                dflistSus.append(Concentracion(prSus,'suscriptores',periodo))
            ConcSus=pd.concat(dflistSus).fillna(1.0).reset_index().drop('index',axis=1)
           
            if select_variable == "Suscriptores":
                colsconSus=ConcSus.columns.values.tolist()
                value1= len(colsconSus)-1 if len(colsconSus)-1 >1 else 2 
                conc=st.slider('Seleccionar número de expresas ',1,value1,1,1)
                fig3 = PlotlyConcentracion(ConcSus) 
                st.write(ConcSus.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconSus[conc]]))
                st.plotly_chart(fig3,use_container_width=True)  
                
        if select_indicador == 'IHH':
            
            for periodo in PERIODOSSUS:
                prSus=SusdptoTV[(SusdptoTV['departamento']==DPTO)&(SusdptoTV['periodo']==periodo)]
                prSus.insert(3,'participacion',(prSus['suscriptores']/prSus['suscriptores'].sum())*100)
                prSus.insert(4,'IHH',IHH(prSus,'suscriptores'))
                dfSuscriptores3.append(prSus.sort_values(by='participacion',ascending=False))
            SusgroupPart3=pd.concat(dfSuscriptores3)
            IHHSus=SusgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()    
            
            fig5=PlotlyIHH(IHHSus)

            if select_variable == "Suscriptores":
                AgGrid(SusgroupPart3)
                st.plotly_chart(fig5,use_container_width=True)
                st.markdown('#### Visualización departamental del IHH')
                periodoME=st.select_slider('Escoja un periodo para calcular el IHH', PERIODOSSUS,PERIODOSSUS[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSSUS:
                    if SusdptoTV[(SusdptoTV['departamento']==departamento)&(SusdptoTV['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prSus=SusdptoTV[(SusdptoTV['departamento']==departamento)&(SusdptoTV['periodo']==periodoME)]
                        prSus.insert(3,'participacion',Participacion(prSus,'suscriptores'))
                        prSus.insert(4,'IHH',IHH(prSus,'suscriptores'))
                        IHHDpto=prSus.groupby(['id_departamento','departamento'])['IHH'].mean().reset_index()
                        dfMap.append(IHHDpto) 
                IHHMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(IHHMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'IHH'],
                    key_on='feature.properties.DPTO',
                    fill_color='Reds', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='IHH',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','IHH'],
                        aliases=['ID Departamento','Departamento','IHH'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480)
                
        if select_indicador == 'Linda':
            dflistSus2=[];datosSus=[];nempresaSus=[];       
            for periodo in PERIODOSSUS:              
                prSus=SusdptoTV[(SusdptoTV['departamento']==DPTO)&(SusdptoTV['periodo']==periodo)]
                nempresaSus.append(prSus.empresa.nunique())
                dflistSus2.append(Linda(prSus,'suscriptores',periodo))
                datosSus.append(prSus)

            NemphisSus=max(nempresaSus)
     
            dSus=pd.concat(datosSus).reset_index().drop('index',axis=1)
            LindSus=pd.concat(dflistSus2).reset_index().drop('index',axis=1).fillna(np.nan)

            if select_variable == "Suscriptores":
                LindconSus=LindSus.columns.values.tolist()
                if NemphisSus==1:
                    st.write("El índice de linda no está definido para éste departamento pues cuenta con una sola empresa")
                    st.write(dSus)
                elif  NemphisSus==2:
                    col1, col2 = st.columns([3, 1])
                    fig10=PlotlyLinda2(LindSus)
                    col1.write("**Datos completos**")                    
                    col1.write(dSus)  
                    col2.write("**Índice de Linda**")
                    col2.write(LindSus)
                    st.plotly_chart(fig10,use_container_width=True)        
                else:    
                    lind=st.slider('Seleccionar nivel',2,len(LindconSus),2,1)
                    fig10=PlotlyLinda(LindSus)
                    st.write(LindSus.fillna(np.nan).reset_index(drop=True).style.apply(f, axis=0, subset=[LindconSus[lind-1]]))
                    with st.expander("Mostrar datos"):
                        st.write(dSus)                    
                    st.plotly_chart(fig10,use_container_width=True)
                    
        if select_indicador == 'Media entrópica':

            for periodo in PERIODOSSUS:
                prSus=SuscriptoresTV[(SuscriptoresTV['departamento']==DPTO)&(SuscriptoresTV['periodo']==periodo)]
                prSus.insert(4,'media entropica',MediaEntropica(prSus,'suscriptores')[0])
                dfSuscriptores.append(prSus)
            SusgroupPart=pd.concat(dfSuscriptores)
            MEDIAENTROPICASUS=SusgroupPart.groupby(['periodo'])['media entropica'].mean().reset_index()    
        
            #Graficas
            
            fig7=PlotlyMEntropica(MEDIAENTROPICASUS)
            
            if select_variable == "Suscriptores":
                st.write(r"""##### <center>Visualización de la evolución de la media entrópica en el departamento seleccionado</center>""",unsafe_allow_html=True)
                st.plotly_chart(fig7,use_container_width=True)      
                periodoME=st.select_slider('Escoja un periodo para calcular la media entrópica', PERIODOSSUS,PERIODOSSUS[-1])
                MEperiodTableSus=MediaEntropica(SuscriptoresTV[(SuscriptoresTV['departamento']==DPTO)&(SuscriptoresTV['periodo']==periodoME)],'suscriptores')[1] 
                
                dfMap=[];

                for departamento in DEPARTAMENTOSSUS:
                    prSus=SuscriptoresTV[(SuscriptoresTV['departamento']==departamento)&(SuscriptoresTV['periodo']==periodoME)]
                    prSus.insert(4,'media entropica',MediaEntropica(prSus,'suscriptores')[0])
                    prSus2=prSus.groupby(['id_departamento','departamento'])['media entropica'].mean().reset_index()
                    dfMap.append(prSus2)
                SusMap=pd.concat(dfMap).reset_index().drop('index',axis=1)
                colsME=['SIJ','SI','WJ','MED','MEE','MEI','Media entropica'] 
                st.write(MEperiodTableSus.reset_index(drop=True).style.apply(f, axis=0, subset=colsME))
                departamentos_df=gdf.merge(SusMap, on='id_departamento')
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
                MunicipiosME=MEperiodTableSus.groupby(['municipio'])['WJ'].mean().reset_index()
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

        if select_indicador == 'Penetración':
            HogaresDpto=Hogares.groupby(['anno','id_departamento'])['hogares'].sum().reset_index()  
            SusDpto=SusdptoTV[(SusdptoTV['departamento']==DPTO)]
            SusDpto=SusDpto.groupby(['periodo','id_departamento','departamento'])[['suscriptores']].sum().reset_index()
            SusDpto.insert(0,'anno',SusDpto.periodo.str.split('-',expand=True)[0])
            HogaresDpto.id_departamento=HogaresDpto.id_departamento.astype('int64')
            HogaresDpto.anno=HogaresDpto.anno.astype('int64')
            SusDpto.id_departamento=SusDpto.id_departamento.astype('int64')
            SusDpto.anno=SusDpto.anno.astype('int64')
            PenetracionDpto=SusDpto.merge(HogaresDpto, on=['anno','id_departamento'], how='left')
            PenetracionDpto.insert(6,'penetracion',PenetracionDpto['suscriptores']/PenetracionDpto['hogares'])
            PenetracionDpto.penetracion=PenetracionDpto.penetracion.round(3)
            if select_variable=='Suscriptores':
                fig12=PlotlyPenetracion(PenetracionDpto)
                AgGrid(PenetracionDpto[['periodo','departamento','suscriptores','hogares','penetracion']])
                st.plotly_chart(fig12,use_container_width=True)
                
        if select_indicador == 'Dominancia':
            for periodo in PERIODOSSUS:
                prSus=SusdptoTV[(SusdptoTV['departamento']==DPTO)&(SusdptoTV['periodo']==periodo)]
                prSus.insert(3,'participacion',(prSus['suscriptores']/prSus['suscriptores'].sum())*100)
                prSus.insert(4,'IHH',IHH(prSus,'suscriptores'))
                prSus.insert(5,'Dominancia',Dominancia(prSus,'suscriptores'))
                dfSuscriptores4.append(prSus.sort_values(by='participacion',ascending=False))
            SusgroupPart4=pd.concat(dfSuscriptores4)
            SusgroupPart4.participacion=SusgroupPart4.participacion.round(2)
            DomSus=SusgroupPart4.groupby(['periodo'])['Dominancia'].mean().reset_index()    
            
            fig13=PlotlyDominancia(DomSus)

            if select_variable == "Suscriptores":
                AgGrid(SusgroupPart4)
                st.plotly_chart(fig13,use_container_width=True)
                st.markdown('#### Visualización departamental de la dominancia')
                periodoME=st.select_slider('Escoja un periodo para calcular la dominancia', PERIODOSSUS,PERIODOSSUS[-1])
                dfMap=[];
                for departamento in DEPARTAMENTOSSUS:
                    if SusdptoTV[(SusdptoTV['departamento']==departamento)&(SusdptoTV['periodo']==periodoME)].empty==True:
                        pass
                    else:    
                        prSus=SusdptoTV[(SusdptoTV['departamento']==departamento)&(SusdptoTV['periodo']==periodoME)]
                        prSus.insert(3,'participacion',Participacion(prSus,'suscriptores'))
                        prSus.insert(4,'IHH',IHH(prSus,'suscriptores'))
                        prSus.insert(4,'Dominancia',Dominancia(prSus,'suscriptores'))
                        DomDpto=prSus.groupby(['id_departamento','departamento'])['Dominancia'].mean().reset_index()
                        dfMap.append(DomDpto) 
                DomMap=pd.concat(dfMap).reset_index().drop('index',axis=1)              
                departamentos_df=gdf.merge(DomMap, on='id_departamento')

                colombia_map = folium.Map(width='100%',location=[4.570868, -74.297333], zoom_start=5,tiles='cartodbpositron')
                tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
                for tile in tiles:
                    folium.TileLayer(tile).add_to(colombia_map)
                choropleth=folium.Choropleth(
                    geo_data=Colombian_DPTO,
                    data=departamentos_df,
                    columns=['id_departamento', 'Dominancia'],
                    key_on='feature.properties.DPTO',
                    fill_color='Oranges', 
                    fill_opacity=0.9, 
                    line_opacity=0.9,
                    legend_name='Dominancia',
                    #bins=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
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
                        fields=['id_departamento','departamento_y','Dominancia'],
                        aliases=['ID Departamento','Departamento','Dominancia'],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                colombia_map.add_child(NIL)
                colombia_map.keep_in_front(NIL)
                col1, col2 ,col3= st.columns([1.5,4,1])
                with col2:
                    folium_static(colombia_map,width=480)
                   
if select_mercado == 'Telefonía móvil':   
    st.title('Telefonía móvil') 
    Trafico=ReadApiVOZTraf()
    Ingresos=ReadApiVOZIng()
    Abonados=ReadApiVOZAbo()
    Trafico=Trafico[Trafico['trafico']>0]
    Ingresos=Ingresos[Ingresos['ingresos']>0]
    Abonados=Abonados[Abonados['abonados']>0]
    Trafico.insert(0,'periodo',Trafico['anno']+'-T'+Trafico['trimestre'])
    Ingresos.insert(0,'periodo',Ingresos['anno']+'-T'+Ingresos['trimestre'])
    Abonados.insert(0,'periodo',Abonados['anno']+'-T'+Abonados['trimestre'])
    Trafnac=Trafico.groupby(['periodo','empresa','id_empresa'])['trafico'].sum().reset_index()
    Ingnac=Ingresos.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
    Abonac=Abonados.groupby(['periodo','empresa','id_empresa'])['abonados'].sum().reset_index()    
    PERIODOS=Trafico['periodo'].unique().tolist()    
    dfTrafico=[];dfIngresos=[];dfAbonados=[]
    dfTrafico2=[];dfIngresos2=[];dfAbonados2=[]
    dfTrafico3=[];dfIngresos3=[];dfAbonados3=[]

    
    select_indicador = st.sidebar.selectbox('Indicador',['Stenbacka', 'Concentración','IHH','Linda'])
    
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

    select_variable = st.selectbox('Variable',['Tráfico', 'Ingresos','Abonados']) 
    
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
    
            prAb=Abonac[Abonac['periodo']==elem]
            prAb.insert(3,'participacion',Participacion(prAb,'abonados'))
            prAb.insert(4,'stenbacka',Stenbacka(prAb,'abonados',gamma))
            dfAbonados.append(prAb.sort_values(by='participacion',ascending=False)) 
        TrafgroupPart=pd.concat(dfTrafico)
        InggroupPart=pd.concat(dfIngresos)
        AbogroupPart=pd.concat(dfAbonados)

        #Gráficas
        fig1=PlotlyStenbacka(TrafgroupPart)
        fig2=PlotlyStenbacka(InggroupPart)
        fig3=PlotlyStenbacka(AbogroupPart)
        ##           
        
        if select_variable == "Tráfico":
            AgGrid(TrafgroupPart)
            st.plotly_chart(fig1, use_container_width=True)
        if select_variable == "Ingresos":
            AgGrid(InggroupPart)
            st.plotly_chart(fig2, use_container_width=True)
        if select_variable == "Abonados":
            AgGrid(AbogroupPart)
            st.plotly_chart(fig3, use_container_width=True)    
            
    if select_indicador == 'Concentración':
        dflistTraf=[];dflistIng=[];dflistAbo=[]
        
        for elem in PERIODOS:
            dflistTraf.append(Concentracion(Trafnac,'trafico',elem))
            dflistIng.append(Concentracion(Ingnac,'ingresos',elem))
            dflistAbo.append(Concentracion(Abonac,'abonados',elem))
        ConcTraf=pd.concat(dflistTraf).fillna(1.0)
        ConcIng=pd.concat(dflistIng).fillna(1.0)
        ConcAbo=pd.concat(dflistAbo).fillna(1.0)      
                    
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
        if select_variable == "Abonados":
            colsconAbo=ConcAbo.columns.values.tolist()
            conc=st.slider('Seleccione el número de empresas',1,len(colsconAbo)-1,1,1)
            fig6=PlotlyConcentracion(ConcAbo)
            st.write(ConcAbo.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconAbo[conc]]))
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
            prAb=Abonac[Abonac['periodo']==elem]
            prAb.insert(3,'participacion',(prAb['abonados']/prAb['abonados'].sum())*100)
            prAb.insert(4,'IHH',IHH(prAb,'abonados'))
            dfAbonados3.append(prAb.sort_values(by='participacion',ascending=False))
        TrafgroupPart3=pd.concat(dfTrafico3)
        InggroupPart3=pd.concat(dfIngresos3)
        AbogroupPart3=pd.concat(dfAbonados3)
        IHHTraf=TrafgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
        IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
        IHHAbo=AbogroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
        
        ##Gráficas
        
        fig7 = PlotlyIHH(IHHTraf)   
        fig8 = PlotlyIHH(IHHIng)
        fig9 = PlotlyIHH(IHHAbo)  
        
        if select_variable == "Tráfico":
            AgGrid(TrafgroupPart3)
            st.plotly_chart(fig7,use_container_width=True)
        if select_variable == "Ingresos":
            AgGrid(InggroupPart3)
            st.plotly_chart(fig8,use_container_width=True)
        if select_variable == "Abonados":
            AgGrid(AbogroupPart3)
            st.plotly_chart(fig9,use_container_width=True)
            
    if select_indicador == 'Linda':
        dflistTraf2=[];dflistIng2=[];dflistAbo2=[]
        
        for elem in PERIODOS:
            dflistTraf2.append(Linda(Trafnac,'trafico',elem))
            dflistIng2.append(Linda(Ingnac,'ingresos',elem))
            dflistAbo2.append(Linda(Abonac,'abonados',elem))
        LindTraf=pd.concat(dflistTraf2).reset_index().drop('index',axis=1).fillna(np.nan)
        LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
        LindAbo=pd.concat(dflistAbo2).reset_index().drop('index',axis=1).fillna(np.nan)     


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
        if select_variable == "Abonados":
            LindconAbo=LindAbo.columns.values.tolist()            
            lind=st.slider('Seleccionar nivel',2,len(LindconAbo),2,1)
            fig12=PlotlyLinda(LindAbo)
            st.write(LindAbo.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconAbo[lind-1]]))
            st.plotly_chart(fig12,use_container_width=True)                           

if select_mercado == 'Internet móvil':
    st.title('Internet móvil') 
    Trafico=ReadApiIMTraf()
    Ingresos=ReadApiIMIng()
    Accesos=ReadApiIMAccesos()

    Trafico=Trafico[Trafico['trafico']>0]
    Ingresos=Ingresos[Ingresos['ingresos']>0]
    Accesos=Accesos[Accesos['accesos']>0]
    Trafico.insert(0,'periodo',Trafico['anno']+'-T'+Trafico['trimestre'])
    Ingresos.insert(0,'periodo',Ingresos['anno']+'-T'+Ingresos['trimestre'])
    Accesos.insert(0,'periodo',Accesos['anno']+'-T'+Accesos['trimestre'])

    Trafnac=Trafico.groupby(['periodo','empresa','id_empresa'])['trafico'].sum().reset_index()
    Ingnac=Ingresos.groupby(['periodo','empresa','id_empresa'])['ingresos'].sum().reset_index()
    Accnac=Accesos.groupby(['periodo','empresa','id_empresa'])['accesos'].sum().reset_index()    
    PERIODOS=Trafico['periodo'].unique().tolist()    
    dfTrafico=[];dfIngresos=[];dfAccesos=[]
    dfTrafico2=[];dfIngresos2=[];dfAccesos2=[]
    dfTrafico3=[];dfIngresos3=[];dfAccesos3=[]
    
    select_indicador = st.sidebar.selectbox('Indicador',['Stenbacka', 'Concentración','IHH','Linda'])
    
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

    select_variable = st.selectbox('Variable',['Tráfico', 'Ingresos','Accesos']) 
    
    ## Cálculo de los indicadores    
    
    if select_indicador == 'Stenbacka':
        gamma=st.slider('Seleccionar valor gamma',0.0,2.0,0.1)
        for elem in PERIODOS:
            prTr=Trafnac[Trafnac['periodo']==elem]
            prTr.insert(3,'participacion',Participacion(prTr,'trafico'))
            prTr.insert(4,'stenbacka',Stenbacka(prTr,'trafico',gamma))
            dfTrafico.append(prTr.sort_values(by='participacion',ascending=False))            
            
            prIn=Ingnac[Ingnac['periodo']==elem]
            prIn.insert(3,'participacion',Participacion(prIn,'ingresos'))
            prIn.insert(4,'stenbacka',Stenbacka(prIn,'ingresos',gamma))
            dfIngresos.append(prIn.sort_values(by='participacion',ascending=False))
    
            prAc=Accnac[Accnac['periodo']==elem]
            prAc.insert(3,'participacion',Participacion(prAc,'accesos'))
            prAc.insert(4,'stenbacka',Stenbacka(prAc,'accesos',gamma))
                     
            dfAccesos.append(prAc.sort_values(by='participacion',ascending=False)) 
        TrafgroupPart=pd.concat(dfTrafico)
        InggroupPart=pd.concat(dfIngresos)
        AccgroupPart=pd.concat(dfAccesos)

        #Gráficas
        fig1=PlotlyStenbacka(TrafgroupPart)
        fig2=PlotlyStenbacka(InggroupPart)
        fig3=PlotlyStenbacka(AccgroupPart)
        ##           
               
        if select_variable == "Tráfico":
            AgGrid(TrafgroupPart)
            st.plotly_chart(fig1, use_container_width=True)
        if select_variable == "Ingresos":
            AgGrid(InggroupPart)
            st.plotly_chart(fig2, use_container_width=True)
        if select_variable == "Accesos":
            AgGrid(AccgroupPart)
            st.plotly_chart(fig3, use_container_width=True)      

    if select_indicador == 'Concentración':
        dflistTraf=[];dflistIng=[];dflistAcc=[]
        
        for elem in PERIODOS:
            dflistTraf.append(Concentracion(Trafnac,'trafico',elem))
            dflistIng.append(Concentracion(Ingnac,'ingresos',elem))
            dflistAcc.append(Concentracion(Accnac,'accesos',elem))
        ConcTraf=pd.concat(dflistTraf).fillna(1.0)
        ConcIng=pd.concat(dflistIng).fillna(1.0)
        ConcAcc=pd.concat(dflistAcc).fillna(1.0)      
                    
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
        if select_variable == "Accesos":
            colsconAcc=ConcAcc.columns.values.tolist()
            conc=st.slider('Seleccione el número de empresas',1,len(colsconAcc)-1,1,1)
            fig6=PlotlyConcentracion(ConcAcc)
            st.write(ConcAcc.reset_index(drop=True).style.apply(f, axis=0, subset=[colsconAcc[conc]]))
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
            prAc=Accnac[Accnac['periodo']==elem]
            prAc.insert(3,'participacion',(prAc['accesos']/prAc['accesos'].sum())*100)
            prAc.insert(4,'IHH',IHH(prAc,'accesos'))
            dfAccesos3.append(prAc.sort_values(by='participacion',ascending=False))
        TrafgroupPart3=pd.concat(dfTrafico3)
        InggroupPart3=pd.concat(dfIngresos3)
        AccgroupPart3=pd.concat(dfAccesos3)
        IHHTraf=TrafgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
        IHHIng=InggroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
        IHHAcc=AccgroupPart3.groupby(['periodo'])['IHH'].mean().reset_index()
        
        ##Gráficas
        
        fig7 = PlotlyIHH(IHHTraf)   
        fig8 = PlotlyIHH(IHHIng)
        fig9 = PlotlyIHH(IHHAcc)  
        
        if select_variable == "Tráfico":
            AgGrid(TrafgroupPart3)
            st.plotly_chart(fig7,use_container_width=True)
        if select_variable == "Ingresos":
            AgGrid(InggroupPart3)
            st.plotly_chart(fig8,use_container_width=True)
        if select_variable == "Accesos":
            AgGrid(AccgroupPart3)
            st.plotly_chart(fig9,use_container_width=True)
            
    if select_indicador == 'Linda':
        dflistTraf2=[];dflistIng2=[];dflistAcc2=[]
        
        for elem in PERIODOS:
            dflistTraf2.append(Linda(Trafnac,'trafico',elem))
            dflistIng2.append(Linda(Ingnac,'ingresos',elem))
            dflistAcc2.append(Linda(Accnac,'accesos',elem))
        LindTraf=pd.concat(dflistTraf2).reset_index().drop('index',axis=1).fillna(np.nan)
        LindIng=pd.concat(dflistIng2).reset_index().drop('index',axis=1).fillna(np.nan) 
        LindAcc=pd.concat(dflistAcc2).reset_index().drop('index',axis=1).fillna(np.nan)     


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
        if select_variable == "Accesos":
            LindconAcc=LindAcc.columns.values.tolist()            
            lind=st.slider('Seleccionar nivel',2,len(LindconAcc),2,1)
            fig12=PlotlyLinda(LindAcc)
            st.write(LindAcc.reset_index(drop=True).style.apply(f, axis=0, subset=[LindconAcc[lind-1]]))
            st.plotly_chart(fig12,use_container_width=True)                
            


st.write(r"""<hr>""", unsafe_allow_html=True)
with st.expander('Referencias'):
    st.write(r"""
-   Herfindahl, O.C. (1950). Concentration in the U.S. Steel Industry. (Tesis de Doctoral no publicada). Columbia University, New York.

-   Hirschman, A.O. (1945). National power and the structure of foreign trade. *University of California Press*. Berkeley.

-   Linda, R. (1976). Methodology of concentration analysis applied to the study of industries and markets.

-   Lis-Gutiérrez, J. (2013). Medidas de concentración y estabilidad de mercado. Una aplicación para Excel. Superintendencia de Industria y Comercio. Documentos de Trabajo, No. 12.

-   Martinez, O.J. (2017). Concentración o desconcentración del mercado de telefonía móvil de Colombia: Una aproximación. *Revista de Economía del Caribe*, 20, 27-51.

-   Melnik, A., Shy, O., & Stenbacka, R. (2008). Assessing market dominance. *Journal of Economic Behavior & Organization*, 68(1), 63-72. https://doi.org/10.1016/j.jebo.2008.03.010

-   Miller, R. A. (1967). Marginal concentration ratios and industrial profit rates: Some empirical results. *Southern Economic Journal*, XXXIV, pp. 259-267.

-   Stazhkova, P., Kotcofana, T., & Protasov, A. (2017). Concentration indices in analysis of competitive environment: case of Russian banking sector. *In CBU International Conference Proceedings*, 5, 458-464.

-   Unión temporal Econometría - Quantil (2020). Propuesta de batería de indicadores para el análsis de competencia. Contrato CRC No. 109 de 2020.

-   U.S. Department of Justice and The Federal Trade Commission (2010). Horizontal Merger Guidelines. Disponible en: https://www.justice.gov/atr/horizontal-merger-guidelines-08192010 """)