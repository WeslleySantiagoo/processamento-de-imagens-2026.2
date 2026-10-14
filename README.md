# Processamento de Imagens - 2026.2

Repositório das Atividades Extras Classe (LEC) da disciplina de Processamento
de Imagens. Cada atividade é independente e possui scripts Python numerados por
questão (`Q1.py`, `Q2.py` e assim por diante).

## Estrutura

| Pasta | Conteúdo principal |
|---|---|
| `Atividade Extra Classe 1/` | Exercícios introdutórios de NumPy com vetores e matrizes. |
| `Atividade Extra Classe 2/` | Transformações de intensidade, gama, limiar, planos de bits, mosaico e combinação de imagens. |
| `Atividade Extra Classe 3/` | Rotação, ampliação, filtragem espacial, esboço a lápis e Laplaciano. |
| `Atividade Extra Classe 4/` | Transformada de Fourier, filtros passa-baixa, passa-alta e gaussiano no domínio da frequência. |
| `Atividade Extra Classe 5/` | Histogramas, equalização e operações morfológicas em imagens binárias. |

As atividades 2 a 5 guardam as imagens de origem em suas pastas de imagens e
salvam as prévias em `resultados/`. A LEC5 também possui um `Qx.md` para cada
questão, com as respostas e análises pedidas. Os arquivos `LEC*.zip` na raiz
são as entregas compactadas.

## Como executar

Crie e ative um ambiente virtual na raiz do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências declaradas no projeto:

```bash
python3 -m pip install -r requirements.txt
```

Execute uma questão de forma individual a partir da raiz do repositório:

```bash
python3 "Atividade Extra Classe 5/Q7.py"
```

Cada script lê os caminhos definidos no início do próprio arquivo, processa a
imagem e salva ou atualiza a prévia correspondente em `resultados/`. Em um
ambiente com interface gráfica, a prévia também é aberta na tela.

## O que esperar

- Os scripts exibem no terminal medidas simples, como quantidade de pixels,
  intensidades ou número de iterações.
- As imagens processadas ficam em `resultados/Qx_resultados.png`.
- Mudanças na imagem, na semente, nos tamanhos de filtros ou nos elementos
  estruturantes alteram naturalmente os resultados.
- A Q7 da LEC5 mantém uma configuração alternativa comentada para comparar o
  preenchimento de buracos em duas imagens.

## Cuidados

- Não renomeie ou mova imagens de entrada sem também ajustar o caminho no
  script correspondente.
- Não substitua as implementações por funções prontas que resolvam todo o
  exercício, como equalização, erosão, dilatação ou filtros prontos.
- Não espere que as prévias sejam idênticas se parâmetros ou imagens forem
  alterados.
- Execute cada questão separadamente; elas não formam uma aplicação única.
