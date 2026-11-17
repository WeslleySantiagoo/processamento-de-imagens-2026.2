# Projeto: Leucograma Digital (Processamento de Imagens 2026.2)
**Professor:** Lucas Fernando da Silva Cambuim  
**Aluno:** Weslley Santiago  
**Curso:** Sistemas de Informação - UFRPE

---

## 1. Visão Geral
Este projeto automatiza a identificação, segmentação fina e classificação diferencial das cinco linhagens leucocitárias (**Neutrófilos, Linfócitos, Monócitos, Eosinófilos e Basófilos**) a partir de fotos de microscopia de lâminas de sangue periférico.

O pipeline foi construído com base nas formulações matemáticas da disciplina (González & Woods):
- **Cores & Intensidade:** Conversão manual vetorizada de RGB para o espaço HSI (Matiz, Saturação e Intensidade) e realce de contraste.
- **Filtragem Espacial:** Atenuação de ruídos espúrios com filtro Gaussiano.
- **Morfologia Matemática:** Operações de abertura, fechamento e preenchimento condicional de buracos para isolamento do núcleo cromatínico e citoplasma.
- **Varredura em Campo Amplo:** Detecção automática de múltiplos leucócitos espalhados em lâminas completas cercadas por hemácias (BCCD).
- **Vetorização com Parquet:** Extração de descritores geométricos ($R_{N/C}$, circularidade, momentos de Hu), cromáticos e de textura, persistidos em `dados/caracteristicas.parquet`.
- **Classificação:** Modelos de Distância Mínima de Protótipos e k-Vizinhos Mais Próximos (k-NN) vetorizados com NumPy e SciPy, comparados ao Scikit-Learn.

---

## 2. Estrutura de Diretórios
```
Projeto/
├── dados/
│   ├── laminas_completas/        # Lâminas de campo amplo para varredura (BCCD)
│   ├── celulas_treino/           # Amostras das 5 linhagens para calibração
│   │   ├── neutrofilo/
│   │   ├── linfocito/
│   │   ├── monocito/
│   │   ├── eosinofilo/
│   │   └── basofilo/
│   └── caracteristicas.parquet   # Tabela colunar de características extraídas
├── resultados/                   # Painéis de auditoria visual gerados pelo pipeline
│   ├── auditoria_1_linhagens.png
│   ├── auditoria_2_hsi.png
│   ├── auditoria_3_segmentacao.png
│   ├── auditoria_4_dispersao.png
│   ├── auditoria_5_matriz_confusao.png
│   ├── auditoria_6_lamina_anotada.png
│   └── auditoria_7_laudo_percentual.png
├── pipeline_leucocitos.py        # Módulo de processamento, segmentação e classificação
├── preparar_dados.py             # Script de download e estruturação das bases
├── leucograma_digital.ipynb      # Jupyter Notebook interativo com execução passo a passo
└── README.md                     # Documentação do projeto
```

---

## 3. Como Executar

### 3.1. Ativação do Ambiente Virtual
Certifique-se de que o ambiente virtual na raiz do projeto está ativado:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2. Execução Interativa (Jupyter Notebook)
Abra e execute o notebook `leucograma_digital.ipynb` no VS Code ou via navegador:
```bash
jupyter notebook Projeto/leucograma_digital.ipynb
```
Cada célula do notebook gera e exibe imediatamente um gráfico ou painel de imagem comprovando a etapa correspondente (auditabilidade completa).

### 3.3. Execução Direta via Terminal (Headless)
Você também pode reexecutar todo o pipeline e regenerar os gráficos de auditoria pelo terminal:
```bash
python Projeto/preparar_dados.py
jupyter nbconvert --to notebook --execute Projeto/leucograma_digital.ipynb --inplace
```
Os resultados em alta resolução são salvos automaticamente na pasta `Projeto/resultados/`.
