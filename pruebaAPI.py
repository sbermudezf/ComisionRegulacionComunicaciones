import streamlit as st
import pandas as pd
from urllib.request import urlopen
import json
resource_idvozmovilAB = '3a9c0304-3795-4c55-a78e-079362373b4d'

url_envios_vozmovilAB = 'https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=' + resource_idvozmovilAB + '' \
                       '&fields[]=anno&fields[]=trimestre&fields[]=modalidad'\
                       '&group_by=anno,trimestre,modalidad'\
                       '&sum=abonados' 

responsevozmovilAB = urlopen(url_envios_vozmovilAB + '&limit=10000') # Se obtiene solo un registro para obtener el total de registros en la respuesta
json_envios_vozmovilAB = json.loads(responsevozmovilAB.read())
Total_registrosvozmovilAB = json_envios_vozmovilAB['result']['total']
ResponsevozmovilAB = urlopen(url_envios_vozmovilAB + '&limit=' + str(Total_registrosvozmovilAB))
Abonados_vozmovilAB = json.loads(ResponsevozmovilAB.read())
# Extracción de registros del json
vozmovilAB = pd.DataFrame(Abonados_vozmovilAB['result']['records'])
vozmovilAB['periodo']=vozmovilAB['anno']+'-T'+vozmovilAB['trimestre']
st.write(vozmovilAB)
