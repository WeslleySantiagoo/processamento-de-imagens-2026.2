# Q3 - Importância da fase na Transformada de Fourier

Foram utilizadas as imagens `baboon_monocromatica.png` como imagem A e
`barbara.pgm` como imagem B. As duas possuem dimensão 512 × 512, permitindo a
combinação direta entre magnitude e fase.

Quando foi usada a magnitude de A com a fase de B, a imagem reconstruída ficou
mais parecida com B. Da mesma forma, a combinação de magnitude de B com fase
de A apresentou a estrutura visual de A. Assim, a fase foi a componente que
mais carregou as informações de posição, contornos e formas da imagem.

Ao reconstruir somente com magnitude, a organização espacial foi perdida e a
imagem deixou de se parecer claramente com a original. Já usando apnas a fase
e magnitude constante, ainda foi possível reconhecer estruturas importantes.
Isso reforça que a magnitude descreve a intensidade das frequências, enquanto
a fase preserva a organização espacial dessas frequências.


## Respostas às perguntas

1. **As imagens reconstruídas se parecem mais com A ou com B?**  
   A imagem formada por magnitude de A e fase de B se pareceu mais com B. A imagem formada por magnitude de B e fase de A se pareceu mais com A.

2. **Qual componente parece carregar a estrutura da imagem?**  
   A fase carregou a maior parte da estrutura visual, incluindo a posição dos elementos, os contornos e as formas principais.

3. **O que acontece ao usar apenas magnitude, com fase igual a zero?**  
   A imagem perdeu a organização espacial original. Permaneceram informações de frequência, mas os objetos deixaram de ser reconhecidos com clareza.

4. **O que acontece ao usar apenas fase, com magnitude constante?**  
   Mesmo com intensidade de magnitude constante, os contornos e a estrutura geral ainda puderam ser identificados, embora a imagem apresentasse menor contraste e uma aparência diferente da original.
