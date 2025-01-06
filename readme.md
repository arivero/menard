# README.md

- For the Clojure NatLang parser, see [menard](https://github.com/ekoontz/menard).
- For the Clojure HTMC learning algorithm, see [pierre-menard](https://github.com/hraberg/pierre-menard).
- For the OCR manager, see [Menard](https://github.com/s-k2/Menard).

If you know of any other repositories or libraries named Menard, please feel free to send a pull request.

## Previous work

- https://arxiv.org/abs/2408.03314 contains the now famous promise "However, in the future, we envision that the outputs of applying additional test-time compute can be distilled back into the base LLM, enabling an iterative self-improvement loop that operates on open-ended natural language." 
- It is surely a reference to same author "Learning by Distilling Context" https://arxiv.org/abs/2209.15189
- https://arxiv.org/abs/2412.14964 is an experiment to inject the RAG context into the model. It gets to memorize it, but the augmented model does not work better than base plus RAG extraction.
- Prompt injection works well for characters. Sort of Persona Injection, via the prompt. See https://arxiv.org/abs/2206.11349
- It also works for small operations of in-context editing. https://arxiv.org/abs/2406.11194

- https://arxiv.org/abs/2210.01351 Less is More: Task-aware Layer-wise Distillation for Language Model Compression




## Menard and Large Language Models (LLMs)

Pierre Menard aimed to recreate "Don Quixote" word-for-word.