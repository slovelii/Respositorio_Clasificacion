import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.set_page_config(
    page_title="Modelo Mental: TF-IDF y Similitud Coseno",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Laboratorio de Recuperación de Información: TF-IDF")
st.subheader("Entendiendo el Modelo Mental de Espacio Vectorial en PLN")

st.markdown("""
Esta herramienta demuestra cómo las computadoras comparan textos transformando documentos en **vectores numéricos**. 
- **TF (Term Frequency)**: Mide la frecuencia de una palabra en un documento específico.
- **IDF (Inverse Document Frequency)**: Castiga palabras muy comunes en la colección y premia términos raros o informativos.
- **Similitud Coseno**: Mide el ángulo entre el vector de la consulta y cada documento para determinar relevancia independientemente de la longitud.
""")

st.divider()

# Documentos de ejemplo
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    # Minúsculas
    text = text.lower()
    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming (reducción a la raíz)
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📄 Base de Conocimiento (Corpus)")
    text_input = st.text_area("Documentos (uno por línea):", default_docs, height=160)
    
    st.markdown("### ❓ Consulta de Entrada")
    question = st.text_input("Escribe tu pregunta o término de búsqueda:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### 💡 Consultas de Prueba Recomendadas")
    st.caption("Prueba estas consultas para comparar cómo se comportan los vectores:")
    
    if st.button("¿Dónde juegan el perro y el gato?", use_container_width=True):
        st.session_state.question = "¿Dónde juegan el perro y el gato?"
        st.rerun()
        
    if st.button("¿Qué hacen los niños en el parque?", use_container_width=True):
        st.session_state.question = "¿Qué hacen los niños en el parque?"
        st.rerun()
        
    if st.button("¿Cuándo cantan los pájaros?", use_container_width=True):
        st.session_state.question = "¿Cuándo cantan los pájaros?"
        st.rerun()
        
    if st.button("¿Dónde suena la música alta?", use_container_width=True):
        st.session_state.question = "¿Dónde suena la música alta?"
        st.rerun()
        
    if st.button("¿Qué animal maúlla durante la noche?", use_container_width=True):
        st.session_state.question = "¿Qué animal maúlla durante la noche?"
        st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

st.markdown("<br>", unsafe_allow_html=True)
if st.button("⚙️ Calcular Pesos y Evaluar Similitud", type="primary", use_container_width=True):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento en la base de conocimiento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta para calcular la similitud.")
    else:
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1  # Incluir todas las palabras
        )
        
        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)
        
        st.divider()
        st.markdown("### 📊 1. Representación Vectorial (Matriz TF-IDF)")
        st.caption("Cada columna representa la raíz (*stem*) de una palabra. El valor asignado crece si la palabra es frecuente en el documento pero decae si está presente en todo el corpus.")
        
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # Calcular similitud con la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        # Encontrar mejor respuesta
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        st.markdown("### 🎯 2. Resultado de la Recuperación por Similitud Coseno")
        st.markdown(f"**Consulta evaluada:** *{question}*")
        
        c_res1, c_res2 = st.columns([3, 1])
        
        with c_res1:
            if best_score > 0.01:
                st.success(f"**Documento más relevante (Doc {best_idx+1}):** {best_doc}")
            else:
                st.warning(f"**Documento seleccionado (Baja coincidencia, Doc {best_idx+1}):** {best_doc}")
                
        with c_res2:
            st.metric(label="Puntaje de Similitud", value=f"{best_score:.3f}")

        # Mostrar tabla comparativa de todos los documentos
        st.markdown("#### Comparativa General de Relevancia")
        df_results = pd.DataFrame({
            "Documento": documents,
            "Puntaje Similitud Coseno": similarities
        }).sort_values(by="Puntaje Similitud Coseno", ascending=False)
        
        st.dataframe(
            df_results.style.background_gradient(subset=["Puntaje Similitud Coseno"], cmap="Blues")
            .format({"Puntaje Similitud Coseno": "{:.3f}"}),
            use_container_width=True
        )
