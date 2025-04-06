💡 Como as ferramentas Streamlit, MLflow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines?

Este projeto integra ferramentas modernas de MLOps e ciência de dados para facilitar o desenvolvimento, rastreamento, monitoramento e deploy de modelos de Machine Learning. Abaixo estão os papéis principais de cada ferramenta nos pipelines:

✅ Rastreamento de Experimentos – MLflow + PyCaret
MLflow permite o rastreamento completo dos experimentos com registro automático de:

Hiperparâmetros

Métricas (ex: Acurácia, F1-Score, Log Loss)

Artefatos como modelos treinados, gráficos e arquivos de avaliação

PyCaret se integra com MLflow e registra automaticamente todos os experimentos executados, facilitando a comparação entre diferentes abordagens de modelagem.

⚙️ Funções de Treinamento – PyCaret + Scikit-Learn
PyCaret simplifica o ciclo completo de modelagem com automação de:

Pré-processamento

Seleção e tuning de modelos

Validação cruzada

Scikit-Learn é utilizado para criação de pipelines customizados e aplicação de técnicas avançadas de feature engineering, garantindo flexibilidade e controle total do processo.

📊 Monitoramento da Saúde do Modelo – Streamlit + MLflow
Streamlit é utilizado para criar dashboards interativos e acompanhar:

Métricas de performance dos modelos em tempo real

Comparações entre modelos com visualizações claras (ex: Matriz de Confusão)

MLflow complementa com logs históricos e versões de modelos, facilitando a auditoria e manutenção.

🔄 Atualização de Modelo – MLflow + Pipelines Automáticos
O pipeline verifica e registra novos modelos com MLflow.

Caso um novo modelo tenha performance superior, ele é promovido como o novo modelo de produção.

Todo o histórico de versões é preservado, permitindo rollback se necessário.

🚀 Provisionamento (Deployment) – Streamlit + MLflow
Streamlit serve como interface de inferência interativa para usuários e analistas.

MLflow possibilita servir o modelo como API REST (mlflow serve), permitindo integração com outras aplicações e sistemas.

O uso de modelos registrados garante reprodutibilidade e consistência entre os ambientes de teste e produção.

📌 1. O modelo é aderente a essa nova base? O que mudou entre uma base e outra? Justifique.
Sim, o modelo é aderente à nova base, desde que ela mantenha o mesmo esquema de features esperadas pelo pipeline (mesmas colunas, tipos de dados e pré-processamento aplicado).

✅ O que mudou entre uma base e outra?
Base de treino (histórica): usada para treinar e validar os modelos, com a variável resposta shot_made_flag disponível.

Base de produção (nova): pode ou não conter a variável resposta. Nessa nova base:

Pode haver drift nos dados, ou seja, mudanças nas distribuições das variáveis.

Pode haver novas combinações de variáveis categóricas, ou valores faltantes inesperados.

Pode haver redução de qualidade das features, caso o pipeline não tenha sido aplicado corretamente.

🔎 Para garantir aderência, é importante usar pipelines robustos e versionados, que transformem os dados novos da mesma forma que os dados históricos foram tratados.

🩺 2. Como monitorar a saúde do modelo com e sem a variável resposta?
✅ Com a variável resposta disponível
Quando a variável shot_made_flag está disponível na base de produção (mesmo que com atraso), é possível:

Calcular métricas de performance reais (ex: Acurácia, F1, Log Loss).

Comparar previsões com os valores reais.

Detectar degradação de performance ao longo do tempo (monitoramento contínuo com MLflow e dashboards Streamlit).

Atualizar ou reverter modelos se necessário, com base em performance real.

Ferramentas como MLflow + Streamlit ajudam a visualizar isso com dashboards operacionais.

⚠️ Sem a variável resposta
Quando a variável resposta não está disponível no momento da predição:

Utilizamos monitoramento indireto:

Distribuição das previsões (ex: distribuição dos scores ou classes previstas).

Drift de dados de entrada (ex: Kolmogorov-Smirnov test entre a base de treino e a base nova).

Monitoramento de outliers ou mudanças de perfil no input.

Ferramentas como EvidentlyAI ou código customizado podem calcular métricas de drift, como:

Population Stability Index (PSI)

Jensen-Shannon Distance

Essas técnicas ajudam a antecipar problemas, mesmo sem feedback imediato.

🔄 3. Estratégias de Retreinamento – Reativa e Preditiva
🧯 Estratégia Reativa
Consiste em reagir à degradação de performance do modelo:

É aplicada quando a variável resposta se torna disponível.

Monitoramos métricas como F1, Log Loss, Acurácia no tempo.

Retreinamento é disparado quando o desempenho cai abaixo de um limite aceitável.

Pode ser agendado (ex: mensal) ou baseado em gatilhos (ex: F1 < 0.7).

🟠 Prós: Simples de implementar; baseado em feedback real.
🔵 Contras: A performance pode já estar ruim antes do retreinamento.

🔮 Estratégia Preditiva
Consiste em prever quando será necessário retreinar, mesmo sem a variável resposta:

Baseada em mudanças nos dados de entrada (Data Drift ou Concept Drift).

Utiliza indicadores como:

Mudança no perfil dos dados

Mudança na distribuição das previsões

Alertas baseados em regras (thresholds)

Pode usar modelos auxiliares para detectar anomalias ou instabilidades.

🟢 Prós: Antecipação de problemas; evita queda brusca de performance.
🔴 Contras: Pode gerar falsos positivos e retreinamentos desnecessários.

