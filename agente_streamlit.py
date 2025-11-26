import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
import re

# --- CONFIG ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
plt.switch_backend('Agg')

# Estados (MEMÓRIA PERSISTENTE ADICIONADA)
for key in ['df', 'chat_history', 'graphs', 'messages', 'current_query', 
            'is_conversation_active', 'cluster_labels', 'cluster_cols', 
            'cluster_inertia', 'analises_realizadas']:  # NOVO: memória das análises
    if key not in st.session_state:
        if key == 'analises_realizadas':
            st.session_state[key] = {}  # DICIONÁRIO PARA ARMAZENAR HISTÓRICO
        elif key in ['df', 'cluster_labels', 'cluster_cols', 'cluster_inertia']:
            st.session_state[key] = None
        elif key == 'is_conversation_active':
            st.session_state[key] = True
        else:
            st.session_state[key] = []

# --- HELPER ---
def detectar_separador(content):
    sample = content[:5000]
    for sep in [',', ';', '\t', '|']:
        try:
            lines = sample.split('\n')[:5]
            counts = [len(line.split(sep)) for line in lines if line.strip()]
            if counts and len(set(counts)) == 1 and counts[0] > 1:
                return sep
        except:
            continue
    return ','

def carregar_arquivo(file):
    ext = file.name.split('.')[-1].lower()
    df = None
    
    try:
        if ext in ['csv', 'txt', 'tsv', 'dat']:
            content = file.getvalue().decode('utf-8', errors='ignore')
            sep = detectar_separador(content)
            
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)
                    if df is not None and len(df) > 0:
                        break
                except:
                    continue
        
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        
        elif ext == 'json':
            file.seek(0)
            df = pd.read_json(file)
        
        else:
            return None, f"Formato .{ext} não suportado"
        
        if df is None or len(df) == 0:
            return None, "Arquivo vazio"
        
        df.columns = [f"Col_{i}" if str(col).startswith('Unnamed') else str(col) 
                      for i, col in enumerate(df.columns)]
        
        return df, "Sucesso"
        
    except Exception as e:
        return None, f"Erro: {str(e)[:100]}"

def sugerir_colunas_clustering():
    """Analisa e sugere as melhores colunas para clustering"""
    df = st.session_state['df']
    if df is None:
        return {"erro": "Sem dados"}
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(num_cols) < 2:
        return {"erro": "Necessita 2+ colunas numéricas"}
    
    # Critérios de seleção
    scores = {}
    
    for col in num_cols:
        try:
            data = df[col].dropna()
            if len(data) < 10:
                continue
            
            # 1. Variância (quanto maior, melhor para clustering)
            variance = data.var()
            
            # 2. Coeficiente de variação (dispersão relativa)
            cv = data.std() / data.mean() if data.mean() != 0 else 0
            
            # 3. Proporção de valores únicos (diversidade)
            unique_ratio = data.nunique() / len(data)
            
            # Score composto (normalizado 0-1)
            score = (
                0.4 * min(variance / data.var().max() if data.var().max() > 0 else 0, 1) +
                0.3 * min(cv, 1) +
                0.3 * unique_ratio
            )
            
            scores[col] = {
                "score": float(score),
                "variancia": float(variance),
                "coef_variacao": float(cv),
                "valores_unicos": int(data.nunique())
            }
        except:
            continue
    
    # Ordena por score
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Top 5 colunas
    top_5 = [(col, info) for col, info in ranked[:5]]
    
    result = {
        "recomendacao": [col for col, _ in top_5[:3]],  # Top 3 para usar
        "detalhes": {col: info for col, info in top_5},
        "justificativa": "Baseado em variância, dispersão e diversidade de valores"
    }
    
    # Salva na memória
    st.session_state['analises_realizadas']['sugestao_clustering'] = result
    
    return result


# --- TOOLS ---

def analisar_dados():
    """Análise estatística completa com REGISTRO EM MEMÓRIA"""
    df = st.session_state['df']
    if df is None:
        return {"erro": "Sem dados"}
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols_all = df.select_dtypes(include=['number']).columns.tolist()
    num_cols_final = []
    
    df_len = len(df)
    for col in num_cols_all:
        try:
            unique_count = df[col].nunique()
            if unique_count < 20 or (unique_count / df_len < 0.05 and unique_count < 100):
                cat_cols.append(col)
            else:
                num_cols_final.append(col)
        except:
            num_cols_final.append(col)
    
    stats = {}
    for col in num_cols_final[:10]:
        try:
            s = df[col].describe()
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))).sum()
            
            stats[col] = {
                "min": float(s['min']),
                "max": float(s['max']),
                "mean": float(s['mean']),
                "std": float(s['std']),
                "outliers": int(outliers)
            }
        except:
            continue
    
    result = {
        "linhas": len(df),
        "colunas": len(df.columns),
        "colunas_disponiveis": list(df.columns[:20]),
        "numericas": num_cols_final[:15],
        "categoricas": cat_cols[:10],
        "stats": stats
    }
    
    # SALVA NA MEMÓRIA
    st.session_state['analises_realizadas']['analise_descritiva'] = result
    
    return result

def correlacao(colunas=None):
    """Correlação com REGISTRO EM MEMÓRIA"""
    df = st.session_state['df']
    num_df = df.select_dtypes(include=['number'])
    
    if colunas:
        num_df = num_df[[c for c in colunas if c in num_df.columns]]
    
    if num_df.shape[1] < 2:
        return {"erro": "Necessita 2+ colunas numéricas"}

    corr = num_df.corr()
    
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            pairs.append({
                "var1": corr.columns[i],
                "var2": corr.columns[j],
                "corr": float(corr.iloc[i,j])
            })
    
    pairs.sort(key=lambda x: abs(x['corr']), reverse=True)
    result = {"top_5_correlacoes": pairs[:5]}
    
    # SALVA NA MEMÓRIA
    st.session_state['analises_realizadas']['correlacoes'] = pairs[:5]
    
    return result

def determinar_k_ideal(colunas):
    """Método do Cotovelo com REGISTRO EM MEMÓRIA"""
    df = st.session_state['df']
    
    valid_cols = [c for c in colunas if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 1:
        return {"erro": "Forneça 1+ coluna numérica"}
        
    try:
        X = df[valid_cols].copy().dropna()
        if X.empty:
            return {"erro": "Dados inválidos"}
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        inertia_list = {}
        max_k = min(len(X) - 1, 10)
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X_scaled)
            inertia_list[k] = kmeans.inertia_
        
        result = {
            "sucesso": True,
            "resultados_cotovelo": inertia_list,
            "colunas_usadas": valid_cols
        }
        
        # SALVA NA MEMÓRIA
        st.session_state['cluster_inertia'] = {"colunas": valid_cols, "inertia": inertia_list}
        st.session_state['analises_realizadas']['metodo_cotovelo'] = result
        
        return result
    except Exception as e:
        return {"erro": f"Erro: {str(e)[:100]}"}

def kmeans_clustering(colunas, k, gerar_grafico_automatico=True):
    """K-Means com REGISTRO EM MEMÓRIA e GRÁFICO AUTOMÁTICO"""
    df = st.session_state['df']
    
    valid_cols = [c for c in colunas if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return {"erro": "Forneça 2+ colunas numéricas"}
    
    if not isinstance(k, int) or k < 2 or k > 10:
        return {"erro": "k deve ser entre 2 e 10"}
        
    try:
        X = df[valid_cols].copy().dropna()
        if X.empty:
            return {"erro": "Dados inválidos"}
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        st.session_state['cluster_labels'] = kmeans.fit_predict(X_scaled)
        st.session_state['cluster_cols'] = valid_cols
        
        result = {
            "sucesso": True,
            "clusters_criados": k,
            "colunas_usadas": valid_cols,
            "tamanho_clusters": pd.Series(st.session_state['cluster_labels']).value_counts().to_dict()
        }
        
        # GERA GRÁFICO AUTOMATICAMENTE se tiver 2+ colunas
        if gerar_grafico_automatico and len(valid_cols) >= 2:
            # Usa as 2 primeiras colunas para visualização
            col_x, col_y = valid_cols[0], valid_cols[1]
            
            # Chama a função de gráfico internamente
            grafico_result = grafico('clusters', col_x, col_y)
            
            if 'graph_id' in grafico_result:
                result['graph_id'] = grafico_result['graph_id']
                result['grafico_gerado'] = f"Gráfico ID {grafico_result['graph_id']} gerado automaticamente"
        
        # SALVA NA MEMÓRIA
        st.session_state['analises_realizadas']['clustering'] = result
        
        return result
    except Exception as e:
        return {"erro": f"Erro: {str(e)[:100]}"}

def grafico(tipo, col_x, col_y=None):
    """Gráficos com REGISTRO EM MEMÓRIA"""
    df = st.session_state['df']
    if df is None:
        return {"erro": "Sem dados"}
    
    try:
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if tipo == 'histograma':
            if col_x not in df.columns:
                return {"erro": f"Coluna {col_x} inválida"}
            sns.histplot(data=df, x=col_x, kde=True, ax=ax)
            ax.set_title(f"Distribuição: {col_x}", fontsize=16)
            
        elif tipo == 'dispersao':
            if not col_y or col_x not in df.columns or col_y not in df.columns:
                return {"erro": "Dispersão requer X e Y"}
            sns.scatterplot(data=df, x=col_x, y=col_y, alpha=0.6, ax=ax)
            ax.set_title(f"Dispersão: {col_x} vs {col_y}", fontsize=16)
            
        elif tipo == 'boxplot':
            if col_x not in df.columns:
                return {"erro": f"Coluna {col_x} inválida"}
            sns.boxplot(data=df, y=col_x, ax=ax, color='skyblue')
            ax.set_title(f"Boxplot: {col_x}", fontsize=16)
            
        elif tipo == 'correlacao':
            num_df = df.select_dtypes(include=['number']).iloc[:, :8]
            sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title("Matriz de Correlação", fontsize=16)
            
        elif tipo == 'clusters':
            labels = st.session_state.get('cluster_labels')
            cluster_cols = st.session_state.get('cluster_cols')
            
            if labels is None or col_x not in cluster_cols or col_y not in cluster_cols:
                return {"erro": "Clusters não criados"}
                
            plot_df = df.copy().dropna(subset=cluster_cols)
            plot_df['Cluster'] = labels[:len(plot_df)]
            
            sns.scatterplot(data=plot_df, x=col_x, y=col_y, hue='Cluster', palette='viridis', ax=ax)
            ax.set_title(f"Clusters K-Means: {col_x} vs {col_y}", fontsize=16)

        elif tipo == 'cotovelo':
            inertia_data = st.session_state.get('cluster_inertia')
            if not inertia_data:
                return {"erro": "Execute determinar_k_ideal primeiro"}
            
            k_values = list(inertia_data['inertia'].keys())
            ssd_values = list(inertia_data['inertia'].values())
            
            ax.plot(k_values, ssd_values, marker='o', linestyle='-', color='purple')
            ax.set_title("Método do Cotovelo", fontsize=16)
            ax.set_xlabel("K")
            ax.set_ylabel("Inércia")
            ax.set_xticks(k_values)
            
        else:
            plt.close(fig)
            return {"erro": "Tipo inválido"}
        
        plt.tight_layout()
        
        gid = len(st.session_state['graphs']) + 1
        st.session_state['graphs'].append(fig)
        
        # SALVA NA MEMÓRIA
        if 'graficos_gerados' not in st.session_state['analises_realizadas']:
            st.session_state['analises_realizadas']['graficos_gerados'] = []
        st.session_state['analises_realizadas']['graficos_gerados'].append({
            "id": gid,
            "tipo": tipo,
            "colunas": [col_x, col_y] if col_y else [col_x]
        })
        
        return {"sucesso": True, "graph_id": gid}
        
    except Exception as e:
        if 'fig' in locals():
            plt.close(fig)
        return {"erro": f"Erro: {str(e)[:50]}"}

# NOVA FUNÇÃO CRÍTICA: CONCLUSÕES
def conclusoes():
    """Sintetiza TODAS as análises realizadas na sessão"""
    analises = st.session_state['analises_realizadas']
    
    if not analises:
        return {"conclusao": "Nenhuma análise foi realizada ainda. Execute análises primeiro."}
    
    conclusao = "## Resumo das Análises Realizadas\n\n"
    
    # 1. Análise Descritiva
    if 'analise_descritiva' in analises:
        desc = analises['analise_descritiva']
        conclusao += f"**Dataset:** {desc['linhas']} linhas, {desc['colunas']} colunas\n"
        conclusao += f"**Numéricas:** {len(desc['numericas'])} colunas\n"
        conclusao += f"**Categóricas:** {len(desc['categoricas'])} colunas\n\n"
        
        if desc['stats']:
            conclusao += "**Estatísticas Principais:**\n"
            for col, stats in list(desc['stats'].items())[:3]:
                conclusao += f"- {col}: média={stats['mean']:.2f}, outliers={stats['outliers']}\n"
            conclusao += "\n"
    
    # 2. Correlações
    if 'correlacoes' in analises:
        conclusao += "**Top 3 Correlações:**\n"
        for corr in analises['correlacoes'][:3]:
            forca = "forte" if abs(corr['corr']) > 0.7 else "moderada" if abs(corr['corr']) > 0.4 else "fraca"
            conclusao += f"- {corr['var1']} ↔ {corr['var2']}: {corr['corr']:.3f} ({forca})\n"
        conclusao += "\n"
    
    # 3. Clustering
    if 'clustering' in analises:
        clust = analises['clustering']
        conclusao += f"**Clusterização:** {clust['clusters_criados']} clusters em {clust['colunas_usadas']}\n"
        conclusao += f"**Distribuição:** {clust['tamanho_clusters']}\n\n"
    
    # 4. Método do Cotovelo
    if 'metodo_cotovelo' in analises:
        cotovelo = analises['metodo_cotovelo']
        conclusao += f"**Método do Cotovelo:** Analisado para {cotovelo['colunas_usadas']}\n\n"
    
    # 5. Gráficos
    if 'graficos_gerados' in analises:
        graficos = analises['graficos_gerados']
        conclusao += f"**Visualizações:** {len(graficos)} gráficos gerados\n"
        tipos = {}
        for g in graficos:
            tipos[g['tipo']] = tipos.get(g['tipo'], 0) + 1
        conclusao += f"**Tipos:** {tipos}\n"
    
    return {"conclusao": conclusao}

# --- TOOLS DEFINITION ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analisar_dados",
            "description": "Análise estatística completa do dataset",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "correlacao",
            "description": "Top 5 correlações entre variáveis",
            "parameters": {
                "type": "object",
                "properties": {
                    "colunas": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "determinar_k_ideal",
            "description": "Método do Cotovelo para K-Means",
            "parameters": {
                "type": "object",
                "properties": {
                    "colunas": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["colunas"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kmeans_clustering",
            "description": "Clusterização K-Means",
            "parameters": {
                "type": "object",
                "properties": {
                    "colunas": {"type": "array", "items": {"type": "string"}},
                    "k": {"type": "integer"}
                },
                "required": ["colunas", "k"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grafico",
            "description": "Gera gráficos: histograma, dispersao, boxplot, correlacao, clusters, cotovelo",
            "parameters": {
                "type": "object",
                "properties": {
                    "tipo": {"type": "string", "enum": ["histograma","dispersao","boxplot","correlacao","clusters","cotovelo"]},
                    "col_x": {"type": "string"},
                    "col_y": {"type": "string"}
                },
                "required": ["tipo", "col_x"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sugerir_colunas_clustering",
            "description": "Analisa estatisticamente e sugere as 3 melhores colunas numéricas para clusterização baseado em variância, dispersão e diversidade",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "conclusoes",
            "description": "CRÍTICO: Resume TODAS análises realizadas. Use quando perguntado sobre conclusões, resumo ou síntese.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# --- EXECUTOR MELHORADO ---
def executar_gpt():
    if st.session_state['df'] is None:
        st.error("Carregue dados primeiro")
        return
    
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY não configurada")
        return
    
    status = st.empty()
    status.info(f"🤖 Processando... (Iteração {len(st.session_state['messages'])})")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        system = """Analista EDA conciso e proativo.
Regras:
1. Execute apenas o solicitado
2. Se pedir conclusões/resumo -> use conclusoes()
3. Se pedir sugestão de colunas para clustering -> use sugerir_colunas_clustering()
4. IMPORTANTE: Após criar clusters, SEMPRE ofereça gerar o gráfico automaticamente
5. Respostas ≤100 palavras
6. Cite graph_id dos gráficos
7. SEMPRE responda algo, nunca fique em silêncio"""
        
        if not st.session_state['messages']:
            status.error("Thread vazia")
            return

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}] + st.session_state['messages'],
            tools=TOOLS,
            max_tokens=400,
            temperature=0.3
        )
        
        msg = response.choices[0].message
        
        # DEBUG
        status.info(f"📥 Resposta recebida. Tool calls: {len(msg.tool_calls) if msg.tool_calls else 0}")
        
        # Resposta Final
        if not msg.tool_calls:
            final = msg.content or "Análise concluída."
            
            resp = {"role": "agent", "content": final}
            if st.session_state.get('graph_ids_to_display'):
                resp['graph_ids'] = st.session_state['graph_ids_to_display']
            
            # Adiciona ao histórico
            st.session_state.chat_history.append({"role": "user", "content": st.session_state['current_query']})
            st.session_state.chat_history.append(resp)
            
            # Limpa estado transiente
            st.session_state['messages'] = []
            st.session_state['current_query'] = None
            st.session_state['graph_ids_to_display'] = None
            
            status.success("✅ Resposta gerada!")
            st.rerun()
            return
        
        # Tool Calls
        st.session_state['messages'].append(msg.model_dump())
        
        if st.session_state.get('graph_ids_to_display') is None:
            st.session_state['graph_ids_to_display'] = []
        
        status.info(f"🔧 Executando {len(msg.tool_calls)} ferramenta(s)...")
        
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            
            try:
                args = json.loads(tool_call.function.arguments)
            except:
                args = {}
            
            # DEBUG
            st.caption(f"Chamando: {func_name}")
            
            result = {"erro": "Função desconhecida"}
            
            if func_name == "analisar_dados":
                result = analisar_dados()
            elif func_name == "correlacao":
                result = correlacao(**args)
            elif func_name == "determinar_k_ideal":
                result = determinar_k_ideal(**args)
            elif func_name == "kmeans_clustering":
                result = kmeans_clustering(**args)
                # Captura graph_id se foi gerado automaticamente
                if "graph_id" in result:
                    st.session_state['graph_ids_to_display'].append(result["graph_id"])
            elif func_name == "grafico":
                result = grafico(**args)
                if "graph_id" in result:
                    st.session_state['graph_ids_to_display'].append(result["graph_id"])
            elif func_name == "sugerir_colunas_clustering":
                result = sugerir_colunas_clustering()
            elif func_name == "conclusoes":
                result = conclusoes()
            
            st.session_state['messages'].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)[:900]
            })
        
        status.warning("🔄 Reprocessando com resultados...")
        # Pequeno delay antes do rerun
        import time
        time.sleep(0.5)
        st.rerun()
        
    except Exception as e:
        status.empty()
        st.error(f"❌ Erro: {str(e)}")
        
        # Limpa estado para permitir nova tentativa
        st.session_state['messages'] = []
        st.session_state['current_query'] = None
        st.session_state['graph_ids_to_display'] = None

# --- UI ---
st.title("📊 Agente EDA I2A2")
st.caption("GPT-4o-mini | Memória Persistente + Conclusões")

EXIT_PHRASES = ["obrigado", "tchau", "fim", "encerrar", "parar", "bye"]

def check_exit(query):
    lower = query.strip().lower()
    if lower in EXIT_PHRASES:
        return True
    for phrase in EXIT_PHRASES:
        if re.search(r'\b' + re.escape(phrase) + r'\b', lower):
            return True
    return False

with st.sidebar:
    st.header("📁 Upload")
    file = st.file_uploader("CSV/Excel/JSON", type=["csv","txt","xlsx","xls","json","tsv"])
    
    if file:
        if st.session_state['df'] is None or file.name != st.session_state.get('last_file'):
            for fig in st.session_state.get('graphs', []):
                plt.close(fig)
                
            st.session_state.update({
                'df': None,
                'chat_history': [],
                'messages': [],
                'graphs': [],
                'current_query': None,
                'graph_ids_to_display': [],
                'last_file': file.name,
                'is_conversation_active': True,
                'cluster_labels': None,
                'cluster_cols': None,
                'cluster_inertia': None,
                'analises_realizadas': {}  # LIMPA MEMÓRIA AO TROCAR ARQUIVO
            })
            
            with st.spinner("Processando..."):
                df_result, msg = carregar_arquivo(file)
                
                if df_result is not None:
                    st.session_state['df'] = df_result
                    st.success(f"✅ {file.name}")
                else:
                    st.error(f"❌ {msg}")
                st.rerun()
    
    if st.session_state['df'] is not None:
        st.dataframe(st.session_state['df'].head(3))
        st.caption(f"{len(st.session_state['df'])} × {len(st.session_state['df'].columns)}")
    
    st.divider()
    
    if OPENAI_API_KEY:
        st.success("🔑 API OK")
    else:
        st.error("🔑 Sem API Key")

# Continue processando tool-calling APENAS se houver mensagens pendentes
if st.session_state['messages']:
    with st.spinner("Aguardando agente..."):
        executar_gpt()

if st.session_state['df'] is not None:
    st.header("💬 Chat")
    
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.info(f"👤 {msg['content']}")
        else:
            st.success(f"🤖 {msg['content']}")
            
            if 'graph_ids' in msg:
                for gid in msg['graph_ids']:
                    if gid <= len(st.session_state['graphs']):
                        st.pyplot(st.session_state['graphs'][gid-1])
            st.divider()
    
    if st.session_state.get('is_conversation_active', True):
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                q = st.text_input("Pergunta:", placeholder="Ex: Quais suas conclusões?", key="input")
            with col2:
                st.write("")
                send = st.form_submit_button("🚀 Enviar", type="primary")

            if st.form_submit_button("🗑️ Limpar"):
                for fig in st.session_state['graphs']:
                    plt.close(fig)
                st.session_state.update({
                    'chat_history': [],
                    'messages': [],
                    'graphs': [],
                    'current_query': None,
                    'graph_ids_to_display': [],
                    'is_conversation_active': True,
                    'cluster_labels': None,
                    'cluster_cols': None,
                    'cluster_inertia': None,
                    'analises_realizadas': {}  # LIMPA MEMÓRIA
                })
                st.rerun()
        
        if send and q.strip():
            if check_exit(q.strip()):
                st.session_state.chat_history.append({"role": "user", "content": q.strip()})
                st.session_state.chat_history.append({
                    "role": "agent", 
                    "content": "Análise encerrada! Use 'Limpar' para nova sessão."
                })
                st.session_state['messages'] = []
                st.session_state['is_conversation_active'] = False
                st.rerun()
            else:
                # DEBUG: Mostra o que está sendo processado
                st.info(f"🔍 Processando: {q.strip()}")
                
                st.session_state['current_query'] = q.strip()
                st.session_state['messages'] = [{"role": "user", "content": q.strip()}]
                st.session_state['graph_ids_to_display'] = []
                
                # Chama executar_gpt() diretamente SEM spinner
                executar_gpt()
    else:
        st.warning("Conversa encerrada. Clique em 'Limpar' para reiniciar.")

else:
    st.info("📋 Faça upload de um arquivo")
    
    with st.expander("📚 Exemplos"):
        st.markdown("""
- **"Analise os dados"**
- **"Correlação entre V1 e V2"**
- **"Qual o melhor k para V1, V2 e V3?"**
- **"Clusterize V1 e V2 com 4 clusters"**
- **"Mostre gráfico de clusters"**
- **"Quais suas conclusões?"** ← NOVA FUNCIONALIDADE
- **"Obrigado"** (encerrar)
        """)