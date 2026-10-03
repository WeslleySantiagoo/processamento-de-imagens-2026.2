# Q4 - Filtro passa-baixa ideal

Foi utilizada a imagem `aerial_view.png`. O filtro passa-baixa ideal foi criado
no domínio da frequência com uma máscara circular central. Foram testados os
raios `r = 10` e `r = 60`.

Com `r = 10`, somente uma região pequena de baixas frequências foi preservada.
Por isso, a imagem ficou mais desfocada e vários detalhes da área urbana foram
removidos. Com `r = 60`, a máscara manteve uma região maior do espectro, o que
preservou mais estruturas e texturas da imagem original.

Assim, o raio controla o equilíbro entre suavização e preservação de detalhes:
raios pequenos removem mais detalhes, enquanto raios maiores mantêm mais
informação visual.
