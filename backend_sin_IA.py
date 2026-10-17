# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:04:18 2026

@author: mquevedo
"""

import os
import json
import time
import pickle
import numpy as np
import re
import unicodedata
import faiss
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

BASE_DIR=Path(__file__).resolve().parent
DATOS_DIR=BASE_DIR.parent/"Insumos"

with open(DATOS_DIR/"datos_backend.pkl","rb") as f:
    datos=pickle.load(f)

proyectos=datos["proyectos"]
competencias=datos["competencias"]
catalogo=datos["catalogo"]
historico=datos["historico"]
entidades_nacion_territorio=datos["entidades"]
fragmentos_manual=datos["fragmentos_manual"]
fragmentos_bases_pnd=datos["fragmentos_bases"]
fragmentos_decretos=datos["fragmentos_decretos"]

index_proyectos=datos["index_proyectos"]
index_competencias=datos["index_competencias"]
index_catalogo=datos["index_catalogo"]
index_historico=datos["index_historico"]
index_entidades=datos["index_entidades"]
index_manual=datos["index_manual"]
index_bases_pnd=datos["index_bases"]
index_decretos=datos["index_decretos"]

modelo_sector=joblib.load(DATOS_DIR/"modelo_sector.pkl")
modelo_entidad=joblib.load(DATOS_DIR/"modelo_entidad.pkl")
modelo=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def limpiar_texto(texto):
    texto=str(texto).lower()
    texto=unicodedata.normalize("NFKD",texto)
    texto=texto.encode("ascii","ignore").decode("utf-8")
    texto=re.sub(r"[^a-z0-9 ]"," ",texto)
    texto=re.sub(r"\s+"," ",texto)
    return texto.strip()

def limpiar_entidad(x):
    x=limpiar_texto(x)
    x=re.sub(r"^\d+\s*","",x)
    x=re.sub(r"\bgestion general\b","",x)
    x=re.sub(r"\bde\b"," ",x)
    x=re.sub(r"\bdel\b"," ",x)
    x=re.sub(r"\by\b"," ",x)
    x=re.sub(r"\s+"," ",x)
    return x.strip()

mapa_entidades=entidades_nacion_territorio[["Entidad","Sector","Compentencia"]].copy()
mapa_entidades["Entidad_limpia"]=mapa_entidades["Entidad"].apply(limpiar_entidad)
mapa_entidades=mapa_entidades.drop_duplicates("Entidad_limpia")
entidades_mapa=mapa_entidades.groupby("Entidad_limpia").first()[["Sector","Compentencia"]].to_dict("index")

def buscar_datos_entidad(entidad):
    if not entidad or entidad=="No determinada":
        return {"Nivel":None,"Competencia":None}
    entidad=limpiar_entidad(entidad)
    if entidad in entidades_mapa:
        return {"Nivel":entidades_mapa[entidad]["Sector"],"Competencia":entidades_mapa[entidad]["Compentencia"]}
    coincidencias=[v for k,v in entidades_mapa.items() if entidad in k or k in entidad]
    if coincidencias:
        return {"Nivel":coincidencias[0]["Sector"],"Competencia":coincidencias[0]["Compentencia"]}
    return {"Nivel":None,"Competencia":None}

def predecir_iniciativa(iniciativa):
    texto=limpiar_texto(iniciativa)
    embedding=modelo.encode([texto],normalize_embeddings=True,convert_to_numpy=True)
    probabilidades_sector=modelo_sector.predict_proba(embedding)[0]
    indice_sector=np.argmax(probabilidades_sector)
    sector=modelo_sector.classes_[indice_sector]
    confianza_sector=probabilidades_sector[indice_sector]
    probabilidades_entidad=modelo_entidad.predict_proba(embedding)[0]
    indice_entidad=np.argmax(probabilidades_entidad)
    entidad=modelo_entidad.classes_[indice_entidad]
    confianza_entidad=probabilidades_entidad[indice_entidad]
    return {"Iniciativa":iniciativa,"Sector_Predicho":sector,"Probabilidad_Sector":round(float(confianza_sector),4),"Entidad_Predicha":entidad,"Probabilidad_Entidad":round(float(confianza_entidad),4)}

def buscar_manual(iniciativa,top_k=5):
    consulta=modelo.encode([limpiar_texto(iniciativa)],normalize_embeddings=True,convert_to_numpy=True)
    similitudes,indices=index_manual.search(consulta,top_k)
    return [{"documento":"Manual de clasificación programático del gasto público.pdf","texto":fragmentos_manual[i],"similitud":float(similitudes[0][j])} for j,i in enumerate(indices[0])]

def buscar_bases_pnd(iniciativa,top_k=5):
    consulta=modelo.encode([limpiar_texto(iniciativa)],normalize_embeddings=True,convert_to_numpy=True)
    similitudes,indices=index_bases_pnd.search(consulta,top_k)
    return [{"documento":"Bases del Plan Nacional de Desarrollo - líneas estratégicas.pdf","texto":fragmentos_bases_pnd[i],"similitud":float(similitudes[0][j])} for j,i in enumerate(indices[0])]

def buscar_decretos(iniciativa,top_k=5):
    consulta=modelo.encode([limpiar_texto(iniciativa)],normalize_embeddings=True,convert_to_numpy=True)
    similitudes,indices=index_decretos.search(consulta,top_k)
    return [{"documento":fragmentos_decretos[i]["documento"],"texto":fragmentos_decretos[i]["texto"],"similitud":float(similitudes[0][j])} for j,i in enumerate(indices[0])]

def buscar_competencias(iniciativa,top_k=5):
    consulta=modelo.encode([limpiar_texto(iniciativa)],normalize_embeddings=True,convert_to_numpy=True)
    similitudes,indices=index_competencias.search(consulta,top_k)
    return [{"Sector":competencias.iloc[i]["Sector"],"Entidad":competencias.iloc[i]["Entidad_Nal_1"],"Competencia":competencias.iloc[i]["Competencia"],"Similitud":float(similitudes[0][j])} for j,i in enumerate(indices[0])]

def buscar_proyectos(iniciativa,top_k=5):
    consulta=modelo.encode([limpiar_texto(iniciativa)],normalize_embeddings=True,convert_to_numpy=True)
    similitudes,indices=index_proyectos.search(consulta,top_k)
    return [{"Proyecto":proyectos.iloc[i]["NombreProyecto"],"Objetivo":proyectos.iloc[i]["ObjetivoGeneral"],"Sector":proyectos.iloc[i]["Sector"],"Entidad":proyectos.iloc[i]["EntidadResponsable"],"Similitud":float(similitudes[0][j])} for j,i in enumerate(indices[0])]

def buscar_catalogo(iniciativa, top_k=5):
    consulta = modelo.encode([limpiar_texto(iniciativa)], normalize_embeddings=True, convert_to_numpy=True)
    similitudes, indices = index_catalogo.search(consulta, top_k)
    resultados = []
    for j, i in enumerate(indices[0]):
        if i < 0 or i >= len(catalogo):
            continue
        fila = catalogo.iloc[i]
        resultados.append({"Numero_eje": fila["Número eje"], "Nombre_eje": fila["Nombre eje"], "Numero_tema": fila["Número tema"], "Nombre_tema": fila["Nombre tema"], "Subtema": fila["Subtema"], "Nombre_subtema": fila["Nombre sub tema"], "Similitud": float(similitudes[0][j])})
    return resultados

def buscar_historico(iniciativa,top_k=5):
    consulta=modelo.encode([limpiar_texto(iniciativa)],normalize_embeddings=True,convert_to_numpy=True)
    similitudes,indices=index_historico.search(consulta,top_k)
    return [{"Iniciativa":historico.iloc[i]["Nombre de iniciativa"],"Sector":historico.iloc[i]["Sector"],"Entidad":historico.iloc[i]["Código y Nombre Entidad ejecutora (tal como aparece en SIIF)"],"Similitud":float(similitudes[0][j])} for j,i in enumerate(indices[0])]

def buscar_entidades(iniciativa,top_k=5):
    consulta=modelo.encode([limpiar_texto(iniciativa)],normalize_embeddings=True,convert_to_numpy=True)
    similitudes,indices=index_entidades.search(consulta,top_k)
    return [{"Sector":entidades_nacion_territorio.iloc[i]["Sector"],"Entidad":entidades_nacion_territorio.iloc[i]["Entidad"],"Competencia":entidades_nacion_territorio.iloc[i]["Compentencia"],"Similitud":float(similitudes[0][j])} for j,i in enumerate(indices[0])]

def obtener_evidencia_completa(iniciativa,top_k=5):
    return {"Supervisado":predecir_iniciativa(iniciativa),"Competencias":buscar_competencias(iniciativa,top_k),"Proyectos":buscar_proyectos(iniciativa,top_k),"Historico":buscar_historico(iniciativa,top_k),"Entidades":buscar_entidades(iniciativa,top_k),"Catalogo":buscar_catalogo(iniciativa,top_k),"Manual":buscar_manual(iniciativa,top_k),"Bases_PND":buscar_bases_pnd(iniciativa,top_k),"Decretos":buscar_decretos(iniciativa,top_k)}

def construir_contexto_final(evidencia):
    return f"""
INICIATIVA:
{evidencia["Supervisado"]["Iniciativa"]}

PREDICCIÓN MODELO SUPERVISADO:
{json.dumps(evidencia["Supervisado"],ensure_ascii=False)}

COMPETENCIAS:
{json.dumps(evidencia["Competencias"],ensure_ascii=False)}

PROYECTOS CANDIDATOS:
{json.dumps(evidencia["Proyectos"],ensure_ascii=False)}

HISTÓRICO:
{json.dumps(evidencia["Historico"],ensure_ascii=False)}

ENTIDADES NACIÓN-TERRITORIO:
{json.dumps(evidencia["Entidades"],ensure_ascii=False)}

CATÁLOGO PND - CANDIDATOS:
{json.dumps(evidencia["Catalogo"],ensure_ascii=False)}

MANUAL:
{json.dumps(evidencia["Manual"],ensure_ascii=False)}

BASES DEL PND:
{json.dumps(evidencia["Bases_PND"],ensure_ascii=False)}

DECRETOS SECTORIALES:
{json.dumps(evidencia["Decretos"],ensure_ascii=False)}
"""

def seleccionar_mejor_resultado(resultados):
    if not resultados:
        return None

    mejor=None
    mejor_similitud=-1

    for resultado in resultados:
        similitud=resultado.get("Similitud",resultado.get("similitud",resultado.get("Score",resultado.get("score",0))))
        try:
            similitud=float(similitud)
        except:
            similitud=0

        if similitud>mejor_similitud:
            mejor_similitud=similitud
            mejor=resultado

    return mejor


def clasificar_sin_ia(iniciativa):
    evidencia=obtener_evidencia_completa(iniciativa,top_k=5)
    supervisado=evidencia["Supervisado"]

    print("SUPERVISADO:")
    print(supervisado)
    sector=supervisado.get("Sector_Predicho","No determinado")
    prob_sector=supervisado.get("Probabilidad_Sector",0)
    
    entidad_modelo=supervisado.get("Entidad_Predicha","No determinada")
    prob_entidad=supervisado.get("Probabilidad_Entidad",0)
    
    candidatos_proyectos=evidencia["Proyectos"]
    mejor_proyecto=seleccionar_mejor_resultado(candidatos_proyectos)

    candidatos_catalogo=evidencia["Catalogo"]
    mejor_catalogo=seleccionar_mejor_resultado(candidatos_catalogo)

    candidatos_entidades=evidencia["Entidades"]

    entidades=[]

    if entidad_modelo and entidad_modelo!="No determinada":
        entidades.append({"Entidad":entidad_modelo,"Score":float(prob_entidad)})

    for candidato in candidatos_entidades:
        entidad=candidato.get("Entidad",candidato.get("Nombre",""))
        similitud=candidato.get("Similitud",candidato.get("similitud",candidato.get("Score",0)))

        if entidad:
            try:
                similitud=float(similitud)
            except:
                similitud=0

            entidades.append({"Entidad":entidad,"Score":similitud})

    entidades_ordenadas=[]

    for entidad in sorted(entidades,key=lambda x:x["Score"],reverse=True):
        nombre=entidad["Entidad"]

        if not any(nombre==x["Entidad"] for x in entidades_ordenadas):
            entidades_ordenadas.append(entidad)

    entidad_1=entidades_ordenadas[0]["Entidad"] if len(entidades_ordenadas)>0 else "No determinada"
    entidad_2=entidades_ordenadas[1]["Entidad"] if len(entidades_ordenadas)>1 else "No determinada"

    nivel_1=buscar_datos_entidad(entidad_1)["Nivel"]
    nivel_2=buscar_datos_entidad(entidad_2)["Nivel"]

    proyecto="No determinado"

    if mejor_proyecto:
        proyecto=mejor_proyecto.get("NombreProyecto",mejor_proyecto.get("Proyecto","No determinado"))
    
    numero_eje = "No determinado"
    nombre_eje = "No determinado"
    numero_tema = "No determinado"
    nombre_tema = "No determinado"
    subtema = "No determinado"
    nombre_subtema = "No determinado"
        
    confianza_sector=float(prob_sector) if prob_sector else 0

    confianza_entidad=float(prob_entidad) if prob_entidad else 0

    confianza_proyecto=0

    if mejor_proyecto:
        confianza_proyecto=mejor_proyecto.get("Similitud",mejor_proyecto.get("similitud",mejor_proyecto.get("Score",0)))

    confianza_catalogo=0
    if mejor_catalogo:
        numero_eje=mejor_catalogo.get("Numero_eje","No determinado")
        nombre_eje=mejor_catalogo.get("Nombre_eje","No determinado")
        numero_tema=mejor_catalogo.get("Numero_tema","No determinado")
        nombre_tema=mejor_catalogo.get("Nombre_tema","No determinado")
        subtema=mejor_catalogo.get("Subtema","No determinado")
        nombre_subtema=mejor_catalogo.get("Nombre_subtema","No determinado")
        confianza_catalogo=mejor_catalogo.get("Similitud",0)
        
    try:
        confianza_proyecto=float(confianza_proyecto)
    except:
        confianza_proyecto=0

    try:
        confianza_catalogo=float(confianza_catalogo)
    except:
        confianza_catalogo=0

    confianza_global=np.mean([confianza_sector,confianza_entidad,confianza_proyecto,confianza_catalogo])

    return {
        "Sector":{"Valor":sector,"Justificacion":f"Sector seleccionado mediante el modelo supervisado con una probabilidad de {confianza_sector:.3f}."},
        "Entidad_1":{"Valor":entidad_1,"Justificacion":"Entidad principal seleccionada mediante la combinación de la predicción supervisada y la evidencia semántica recuperada."},
        "Nivel_Entidad_1":nivel_1,
        "Entidad_2":{"Valor":entidad_2,"Justificacion":"Segunda entidad seleccionada entre las entidades pertinentes recuperadas mediante similitud semántica."},
        "Nivel_Entidad_2":nivel_2,
        "Proyecto":{"Valor":proyecto,"Justificacion":"Proyecto seleccionado por presentar la mayor similitud semántica entre los proyectos recuperados."},
        "Numero_eje":{"Valor":numero_eje,"Justificacion":"Número del eje correspondiente al registro del Catálogo PND con mayor similitud semántica."},
        "Nombre_eje":{"Valor":nombre_eje,"Justificacion":"Nombre del eje seleccionado a partir del registro del Catálogo PND con mayor similitud semántica."},
        "Numero_tema":{"Valor":numero_tema,"Justificacion":"Número del tema correspondiente al registro seleccionado del Catálogo PND."},
        "Nombre_tema":{"Valor":nombre_tema,"Justificacion":"Nombre del tema correspondiente al registro seleccionado del Catálogo PND."},
        "Subtema":{"Valor":subtema,"Justificacion":"Código o identificador del subtema correspondiente al registro seleccionado del Catálogo PND."},
        "Nombre_subtema":{"Valor":nombre_subtema,"Justificacion":"Nombre del subtema correspondiente al registro seleccionado del Catálogo PND."},
        "ConclusionGeneral":"Clasificación realizada mediante modelo supervisado, recuperación semántica y reglas determinísticas, sin utilización de IA generativa.",
        "ConfianzaGlobal":float(confianza_global)
    }

def clasificar_iniciativa_completa(iniciativa):
    inicio = time.time()
    clasificacion = clasificar_sin_ia(iniciativa)
    return {
        "Iniciativa": iniciativa,
        "Clasificacion": clasificacion,
        "Tiempo_Segundos": time.time() - inicio}
