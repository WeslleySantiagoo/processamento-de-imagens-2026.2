# Q2 — Reconstrução da imagem

Foi utilizada a imagem `butterfly.png`. Primeiro, foi calculada a Transformada de
Fourier 2D da imagem. Em seguida, a transformada inversa foi aplicada usando o
espectro completo, ou seja, mantendo magnitude e fase originais.

A imagem reconstruída ficou visualmente igual à original. O erro máximo foi de
aproximadamente `0,0000305` e o erro quadrático médio ficou próximo de zro.
Essa diferença muito pequena ocorre por limitações numéricas dos cálculos com
ponto flutuante.

Também foram feitas duas reconstruções parciais. Ao usar apenas a magnitude,
a organização espacial da borboleta deixou de ser preservada com clareza. Já na
reconstrução usando apenas a fase e magnitude constante, contornos e estruturas
importantes ainda puderam ser percebidos. Assim, nesta questão, a fase foi a
componente que mais preservou a estrutura visual da imagem.
