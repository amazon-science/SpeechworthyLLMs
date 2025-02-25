# Speechworthy Instruction-tuned Language Models

This repository contains the code and data accompanying the paper "Speechworthy Instruction-tuned Language Models" by Hyundong Cho et al., presented at EMNLP 2024. The work focuses on enhancing instruction-tuned language models (ITLMs) to generate responses better suited for speech-based applications.

## Overview

Current ITLMs are primarily trained using textual preference data, which may not align with the unique requirements of speech modalities. To address this, our research explores:

1. **Prompting Strategies**: Utilizing techniques grounded in radio-industry best practices to guide models in producing speech-friendly outputs.
2. **Preference Learning**: Developing a novel speech-based preference dataset, **SPEECHPREF**, comprising 20,000 samples. These samples were generated using diverse prompts to induce varying degrees of speech suitability and were labeled by annotators who listened to response pairs.

Our findings indicate that both prompting and preference learning independently enhance the speech suitability of ITLMs. Moreover, combining these methods yields the best results, with responses preferred or tied to the base model in 76.2% of comparisons on average.

For a comprehensive understanding, please refer to the full paper: [Speechworthy Instruction-tuned Language Models](https://aclanthology.org/2024.emnlp-main.595/).

## Repository Contents

- `data/`: Contains the **SPEECHPREF** dataset used for preference learning.
- `code/`: Source code for:
  - Implementing prompting strategies.
  - Training models with speech-based preference data.
  - Evaluation scripts for assessing speech suitability.

## Citation

If you find our work or reuse our data, please cite our paper:
```
@inproceedings{cho-etal-2024-speechworthy,
    title = "Speechworthy Instruction-tuned Language Models",
    author = "Cho, Hyundong Justin  and
      Jedema, Nicolaas Paul  and
      Ribeiro, Leonardo F. R.  and
      Sharma, Karishma  and
      Szekely, Pedro  and
      Moschitti, Alessandro  and
      Janssen, Ruben  and
      May, Jonathan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.595/",
    doi = "10.18653/v1/2024.emnlp-main.595",
    pages = "10652--10670",
    abstract = "Current instruction-tuned language models are exclusively trained with textual preference data and thus may not be aligned to the unique requirements of other modalities, such as speech. To better align language models with the speech domain, we explore i) prompting strategies based on radio-industry best practices and ii) preference learning using a novel speech-based preference data of 20K samples collected by annotators who listen to response pairs. Both human and automatic evaluation show that both prompting and preference learning increase the speech-suitability of popular instruction tuned LLMs. More interestingly, we show that these methods are additive; combining them achieves the best win rates in head-to-head comparison, resulting in responses that are preferred or tied to the base model in 76.2{\%} of comparisons on average. Lastly, we share lexical, syntactical, and qualitative analyses that elicit how our studied methods differ with baselines in generating more speech-suitable responses."
}
```


## License
The data and code use differnt licenses; code is released under CC BY NC 4.0, while data is realesed under CC by SA; see the LICENSE.txt in both repos for the exact license terms. This software is release as-is with no warranty.
