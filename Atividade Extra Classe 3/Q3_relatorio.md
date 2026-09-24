# Q3 — Relatório de filtragem de imagens

## Objetivo

Nesta questão, foi feita a aplicação de filtros de suavização na imagem. A ideia principal foi observar como a escolha do kernel influencia a redução de ruídos, a preservação de detalhes e a aparência das bordas.

## Como a filtragem foi implementada

A filtragem foi realizada por correlação. Em termos práticos, para gerar cada pixel da imagem de saída, o programa considera uma pequena região ao redor do pixel correspondente na imagem original e combina seus valores usando os pesos definidos no kernel.

A solução foi implementada sem usar funções prontas de filtragem. Primeiro, a imagem foi expandida nas bordas pela repetição dos pixels mais próximos. Isso foi necessário para que a imagem filtrada mantivesse o mesmo tamanho da original. Em seguid, as janelas locais foram formadas e combinadas vetorialmente com os pesos de cada máscara.

No final, os valores calculados foram arredondados e limitdos ao intervalo `[0, 255]`, que é a faixa válida para imagens em tons de cinza de 8 bits.

## Efeito dos filtros aplicados

| Filtro | Efeito percebido |
|---|---|
| Caixa 3 * 3 | Produz uma suavização leve. Pequenas variações de intensidade e ruídos são reduzidos, mas a maior parte das estruturas da imagem ainda pode ser percebida com facilidade. |
| Caixa 5 * 5 | O desfoque fica mais evidente. Detalhes menores das construções, vias e texturas da imagem aérea começam a perder definição. |
| Caixa 7 * 7 | Gera a suavização mais forte entre os filtros de caixa. Apesar de reduzir bastante as variações locais, também deixa bordas e detalhes finos menos nítidos. |
| Gaussiano 3 * 3 | Suaviza a imagem de forma leve, dando mais importância aos pixels próximos do centro. Por isso, preservou melhor os contornos do que o fltro de caixa 3 * 3. |
| Gaussiano 5 * 5 | Produz um efeito de desfoque mais gradual. A imagem fica suavizada, mas as transições entre regiões diferentes permanecem visualmente mais naturais. |
| Gaussiano 7 * 7 | Aplica a suavização mais intensa entre os gaussianos. Reduz detalhes de pequena escala, mas preservou melhor a aparência geral das bordas em comparação com a caixa 7 * 7. |

## Comparação entre filtros de caixa e gaussianos

A principal diferença entre os dois tipos de filtro está nos pesos usados dentro do kernel. No filtro de caixa, todos os pixels da vizinhança têm o mesmo peso. Dessa forma, o resultado corresponde a uma média simples da região analisada.

Já no filtro gaussiano, os pixels mais próximos do centro recebem pesos maiores e os mais distantes influenciam menos no resultado. Na prática, isso gerou uma suavização mais equilibrada, pois a informação do pixel central é mais preservada.

Também foi possível perceber que, ao aumentar o tamanho do kernel, o efeito de desfoque aumenta. Esse comportamento é esperado, uma vizinhança maior mistura mais informações e reduz mais variações locais. Em contrapartida, detalhes importantes da imagem também foram perdidos.

## Testes realizados

Foram feitos alguns testes simples para verificar se a implementação da correlação estava funcionando corretamente:

- A aplicação de um kernel identidade devolveu a mesma matriz usada como entrada.
- Uma matriz com todos os pixels de mesma intensidade permaneceu constante depois da aplicação de um filtro de caixa normalizado.
- Foi verificado que os kernels de caixa têm soma igual a `1`, o que ajuda a preservar a intensidade média em regiões uniformes.

## Limitações

O tratamento de bordas por replicação foi escolhido para manter o tamanho da imagm. No entanto, essa escolha(na minha visão) causou pequenas diferenças nas regiões próximas às extremidades. Além disso, os filtros de suavização não conseguem separar automaticamente ruído de detalhes importantes, por isso, um filtro mais forte pode remover ruído, mas também reduzir informações relevantes da imagem.
