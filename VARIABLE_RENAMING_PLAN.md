# DCoDi Debiasing - Plano de Renomeação de Variáveis

Este documento mapeia todas as variáveis que serão renomeadas para tornar o código mais profissional e legível para a publicação do artigo.

## 1. VARIÁVEIS PRINCIPAIS (decodi_main.py)

### Argumentos de Linha de Comando
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `acceptable_modes` | `acceptable_modes` | mantem |
| `mode` | `mode_target` | troca apenas para o mode_target |
| `enable_safety_guidance` | `enable_dcodi_debiasing` | Relaciona diretamente ao método DCoDi |
| `default` | `use_default_hyperparameters` | Mais explícito sobre o que controla |

### Configuração de Modelos
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `base_dir` | `project_root_dir` | Mais claro sobre sua função |
| `cache_dir` | `model_cache_dir` | Especifica que é para cache de modelos |
| `base` | `sdxl_base_pipeline` | Mais descritivo |
| `refiner` | `sdxl_refiner_pipeline` | Mais descritivo |

### Prompts e Conceitos de Bias
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `prompt` | `generation_prompt` | Mais específico |
| `safety_text_concept` | `bias_concept_description` | Relaciona ao conceito de bias sendo tratado |

### Hiperparâmetros do DCoDi (antigo SLD)
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `sld_guidance_scale` | `dcodi_guidance_scale` | Renomeia de SLD para DCoDi |
| `sld_threshold` | `dcodi_bias_threshold` | Mais específico sobre detecção de bias |
| `sld_warmup_steps` | `dcodi_warmup_steps` | Consistente com a nomenclatura |
| `sld_momentum_scale` | `dcodi_momentum_scale` | Consistente com a nomenclatura |
| `sld_mom_beta` | `dcodi_momentum_beta` | Mais claro e consistente |

### Hiperparâmetros de Geração
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `num_inference_steps` | `num_inference_steps` | num_inference_steps |
| `denoising_ratio` | `denoising_ratio` | Especifica que é a proporção do pipeline base |
| `guidance_scale` | `cfg_guidance_scale` | Especifica que é Classifier-Free Guidance |
| `do_classifier_free_guidance` | `use_classifier_free_guidance` | Mais claro |
| `output_type` | `output_type` | Mais específico |
| `freeze_random_latents` | `freeze_random_latents` | Mais intuitivo |
| `text_encoder_lora_scale` | `text_encoder_lora_scale` | Mais conciso |
| `refiner_device` | `refiner_device` | Mais específico |
| `denoising_start` | `denoising_start` | Mais claro sobre sua função |

### Processamento de Imagens
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `rnd_list` | `random_seed_list` | Mais descritivo |
| `rnd` | `current_seed` | Mais claro no contexto do loop |
| `original_imgs_list` | `generated_images_metadata` | Mais descritivo do conteúdo |
| `original_img` | `generated_image` | Evita confusão com "original" |
| `pickle_files_dir` | `data_files_dir` | Mais genérico |
| `pickle_file_path` | `seed_list_file_path` | Mais específico |
| `images_dir` | `output_images_dir` | Mais claro |
| `mode_dir` | `profession_output_dir` | Mais descritivo |
| `name_image` | `image_type` | Mais claro (original/debiased) |
| `file_name` | `output_filename` | Mais específico |
| `file_path` | `output_filepath` | Mais específico |
| `is_default` | `hyperparameter_config` | Mais descritivo |

## 2. FUNÇÕES E IMPORTS

### Imports e Nomes de Funções
| Atual | Nova | Justificativa |
|---|---|---|
| `from sdxl_debias_class_free_guidance import sdxl_debias_cfg` | `from dcodi_generation_pipeline import dcodi_generate_image` | Nomenclatura mais consistente |

## 3. VARIÁVEIS NO ARQUIVO DE GERAÇÃO (sdxl_debias_class_free_guidance.py)

### Função Principal
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `sdxl_debias_cfg` | `dcodi_generate_image` | Nome mais descritivo da função |

### Embeddings e Processamento de Texto
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `safety_text_concept` | `bias_concept_text` | Mais claro sobre o propósito |
| `enable_safety_guidance` | `enable_dcodi_debiasing` | Consistente com o método |
| `safety_concept_input` | `bias_concept_tokens` | Mais específico |
| `safety_embeddings` | `bias_concept_embeddings` | Mais claro |
| `safety_concept_input2` | `bias_concept_tokens_2` | Consistente |
| `safety_embeddings2` | `bias_concept_embeddings_2` | Consistente |
| `safety_pooled_embeds` | `bias_concept_pooled_embeddings` | Mais descritivo |
| `safety_embeds` | `bias_concept_combined_embeddings` | Mais específico |

### Predições de Ruído
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `noise_pred_safety_concept` | `noise_pred_bias_concept` | Consistente com a terminologia |
| `noise_guidance_safety` | `dcodi_debiasing_guidance` | Mais específico ao método |
| `safety_momentum` | `dcodi_momentum` | Consistente |
| `safety_concept_scale` | `bias_detection_scale` | Mais claro sobre a função |

### Variáveis Técnicas
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `latent_model_input` | `unet_latent_input` | Mais específico |
| `cutoff_timesteps` | `base_pipeline_timesteps` | Mais descritivo |
| `cutoff_num_inference_steps` | `base_pipeline_steps` | Consistente |

## 4. ARQUIVOS DE PIPELINE BREAKDOWN

### sdxl_pipeline_base_breakdown.py
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `adds_sld` | `prepare_dcodi_embeddings` | Mais descritivo da função |
| `safety_embeds` | `bias_concept_embeddings` | Consistente |
| `safety_pooled_embeds` | `bias_concept_pooled_embeddings` | Consistente |
| `enable_safety_guidance` | `enable_dcodi_debiasing` | Consistente |

### sdxl_pipeline_refiner_breakdown.py
| Variável Atual | Variável Nova | Justificativa |
|---|---|---|
| `refiner_timesteps` | `prepare_refiner_timesteps` | Mais descritivo |
| `refiner_adds` | `prepare_refiner_embeddings` | Mais descritivo |
| `refiner_denoising_loop` | `execute_refiner_denoising` | Mais descritivo |

## 5. CONSTANTES E CONFIGURAÇÕES

### Novas Constantes Sugeridas
```python
# Profissões suportadas
SUPPORTED_PROFESSIONS = ['firefighter', 'nurse', 'ceo']

# Configurações padrão do DCoDi

#aqui ok, mas cuida para nao mudar nenhum valor, apenas os nomes.
DEFAULT_DCODI_CONFIG = {
    'guidance_scale': 2000,
    'bias_threshold': 1.0,
    'warmup_steps': 7,
    'momentum_scale': 0.5,
    'momentum_beta': 0.7
}

CUSTOM_DCODI_CONFIG = {
    'guidance_scale': 15000,
    'bias_threshold': 0.025,
    'warmup_steps': 7,
    'momentum_scale': 0.5,
    'momentum_beta': 0.7
}
```

## 6. COMENTÁRIOS E DOCUMENTAÇÃO

### Seções que Precisam de Novos Comentários
- Substituir "SAFE LATENT DIFFUSION" por "DCODI DEBIASING PARAMETERS"
- Adicionar comentários explicativos sobre o método DCoDi (Todos comentarios devem ser em ingles, e apenas comentarios realmente importantes.)
- Melhorar documentação das funções principais, novamente em ingles e poucos comentarios

## 7. NOMES DE ARQUIVOS (OPCIONAL)

### Sugestões de Renomeação de Arquivos
| Arquivo Atual | Arquivo Sugerido | Justificativa |
|---|---|---|
| `decodi_main.py` | `dcodi_main.py` | Consistência na nomenclatura |
| `sdxl_debias_class_free_guidance.py` | `dcodi_generation_pipeline.py` | Mais descritivo |
| `sdxl_pipeline_base_breakdown.py` | `dcodi_base_pipeline_utils.py` | Mais claro |
| `sdxl_pipeline_refiner_breakdown.py` | `dcodi_refiner_pipeline_utils.py` | Mais claro |

---

## Precisamos gerar um README tambem.
# No README precisamos deixar um espaço para colocarmos aquela famosa imagem do antes e depois
# O README assim como todos comentarios do codigo devem estar em ingles.
# Explique bem no readme a ponto de qualquer pessoa rodar nosso codigo e conseguir rodar ele direito funcionando.
