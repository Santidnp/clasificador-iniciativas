# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:42:30 2026

@author: mquevedo
"""

import streamlit as st
import pandas as pd
import io
from backend_sin_IA import clasificar_iniciativa_completa
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Clasificador de Iniciativas",page_icon="📊",layout="wide")
    
col1, col2, col3 = st.columns([4, 1, 4])

DNP = Image.open(Path(__file__).resolve().parent / "Imagen1.png")

with col1:
    st.write("                                                                                         ")

with col2:
    st.image(DNP, use_container_width =True) 
with col3:
    st.write("                                                            ")


st.title("Clasificación automática de iniciativas")
st.write("Carga un archivo Excel y selecciona la columna que contiene las iniciativas.")

archivo=st.file_uploader("Cargar archivo Excel",type=["xlsx","xls"])

if archivo is not None:
    df=pd.read_excel(archivo)
    st.success(f"Archivo cargado correctamente: {len(df):,} registros")
    columna=st.selectbox("Selecciona la columna de iniciativas",df.columns)

    if st.button("Clasificar iniciativas",type="primary"):
        iniciativas = (df[columna].dropna().astype(str).str.strip())

        iniciativas = iniciativas[iniciativas != ""].drop_duplicates().tolist()
        
        if not iniciativas:
            st.warning("La columna seleccionada no contiene iniciativas para clasificar.")
            st.stop()
    
        total=len(iniciativas)
        st.info(f"Se encontraron {total:,} iniciativas únicas para clasificar.")
        resultados=[]
        errores=[]
        barra=st.progress(0)
        estado=st.empty()

        for i,iniciativa in enumerate(iniciativas,1):
            try:
                estado.write(f"Clasificando {i:,} de {total:,}: {iniciativa[:100]}")
                resultado=clasificar_iniciativa_completa(iniciativa)
                resultados.append(resultado)
            except Exception as e:
                errores.append({"Iniciativa":iniciativa,"Error":str(e)})
            barra.progress(i/total)

        estado.success("Clasificación terminada.")

        filas=[]

        for resultado in resultados:
            c=resultado["Clasificacion"]
            filas.append({
                "Iniciativa":resultado["Iniciativa"],
                "Sector":c["Sector"]["Valor"],
                "Entidad_1":c["Entidad_1"]["Valor"],
                "Nivel_Entidad_1":c["Nivel_Entidad_1"],
                "Entidad_2":c["Entidad_2"]["Valor"],
                "Nivel_Entidad_2":c["Nivel_Entidad_2"],
                "Proyecto":c["Proyecto"]["Valor"],
                "Numero_eje":c["Numero_eje"]["Valor"],
                "Nombre_eje":c["Nombre_eje"]["Valor"],
                "Numero_tema":c["Numero_tema"]["Valor"],
                "Nombre_tema":c["Nombre_tema"]["Valor"],
                "Subtema":c["Subtema"]["Valor"],
                "Nombre_subtema":c["Nombre_subtema"]["Valor"],
                "Tiempo_Segundos":resultado["Tiempo_Segundos"]})

        resultado_unico=pd.DataFrame(filas)

        resultado_salida=df.copy()
        resultado_salida["__clave_iniciativa"]=resultado_salida[columna].fillna("").astype(str)
        resultado_unico["__clave_iniciativa"]=resultado_unico["Iniciativa"].fillna("").astype(str)

        columnas_resultado=[c for c in resultado_unico.columns if c not in ["Iniciativa","__clave_iniciativa"]]

        resultado_salida=resultado_salida.merge(resultado_unico[["__clave_iniciativa"]+columnas_resultado],on="__clave_iniciativa",how="left")
        resultado_salida=resultado_salida.drop(columns=["__clave_iniciativa"])

        st.subheader("Resultados")
        st.dataframe(resultado_salida,use_container_width=True)

        if errores:
            st.warning(f"Se presentaron {len(errores)} errores.")
            st.dataframe(pd.DataFrame(errores),use_container_width=True)

        buffer=io.BytesIO()

        with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
            resultado_salida.to_excel(writer,index=False,sheet_name="Clasificacion")
            if errores:
                pd.DataFrame(errores).to_excel(writer,index=False,sheet_name="Errores")

        buffer.seek(0)

        st.download_button(
            label="📥 Descargar resultados en Excel",
            data=buffer,
            file_name="Resultado_Iniciativas_Organizado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")