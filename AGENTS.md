# Processamento de Imagens

Disciplina de processamento de imagens. Os exercícios envolvem manipulação de pixels, decodificação de informações em canais de cor (RGB), e operações sobre imagens.

## Bibliotecas permitidas

Usar apenas:
- **numpy** — manipulação de arrays e operações matriciais
- **scipy** — processamento científico (filtros, transformações, etc.)
- **imageio** — leitura e escrita de imagens (`imageio.v2.imread`)

**Não usar**: Pillow/PIL, OpenCV, ou outras bibliotecas de imagem.

## Ambiente

- Python venv em `/Users/caue.lira/Desktop/facul/proc_imagens/venv`
- Executar com: `/Users/caue.lira/Desktop/facul/proc_imagens/venv/bin/python3`

## Estrutura dos exercícios

- Cada exercício fica em uma pasta (`ex1/`, `ex2/`, etc.)
- Contém `README.txt` com o enunciado, `AGENTS.md` com contexto adicional
- Casos de teste: `N.in` (entrada) e `N.out` (saída esperada)
- Testar com `diff` contra os arquivos `.out`
