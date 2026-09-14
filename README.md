# Jogo da Vida de Conway — Implementação Paralela com OpenMP

Trabalho final (1º Bimestre) — Implementação do **Jogo da Vida de Conway**
em **C**, com paralelização utilizando **OpenMP**.

## Regras implementadas

| Estado | Vizinhas vivas | Resultado         |
|--------|-----------------|--------------------|
| Viva   | 0 ou 1          | Morre (isolamento) |
| Viva   | 2 ou 3          | Continua viva      |
| Viva   | 4 ou mais       | Morre (superpopulação) |
| Morta  | 3               | Nasce              |
| Morta  | outros valores  | Continua morta     |

A grade é **toroidal**: as bordas se conectam com o lado oposto, evitando
tratamento especial para células na borda.

## Onde está a paralelização

O cálculo da próxima geração (`proxima_geracao`) percorre todas as células
da grade. Como o novo estado de cada célula depende **apenas da grade
atual** (que não é alterada durante o cálculo), não existe dependência de
dados entre as iterações — cada célula pode ser processada de forma
totalmente independente. Por isso o duplo laço (linhas × colunas) é
paralelizado com:

```c
#pragma omp parallel for collapse(2) schedule(static)
for (int i = 0; i < linhas; i++) {
    for (int j = 0; j < colunas; j++) {
        ...
    }
}
```

A contagem de células vivas (`conta_vivas`), usada apenas para estatísticas
exibidas na tela, também foi paralelizada com `reduction(+:total)`.

## Compilação

```bash
gcc -fopenmp -O2 -o game_of_life game_of_life.c
```

## Execução

```bash
.\game_of_life [rows] [cols] [generations] [num_threads]
```

Exemplo (grade 40x80, 200 gerações, 4 threads):

```bash
.\game_of_life 40 80 200 4
```

Se nenhum argumento for informado, o programa usa uma grade 30x60, 100
gerações e o número máximo de threads disponíveis na máquina.

## Saída

O programa imprime, a cada geração:
- o número da geração;
- a quantidade de células vivas;
- o tempo acumulado de simulação;
- a grade, onde `#` representa célula viva e `.` célula morta.

Ao final, exibe o tempo total de execução, útil para comparar o desempenho
com diferentes quantidades de threads (por exemplo, rodando com
`num_threads = 1` e depois com `num_threads = 4` na mesma grade).

## Estrutura do arquivo

- `game_of_life.c` — código-fonte completo (único arquivo).
