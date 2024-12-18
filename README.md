# yaicon-our-topic-is-blank

<h1 align="center">
  <b>Our topic is blank</b><br>
  <b>우리 팀의 주제는 [ ]입니다 </b><br>
</h1>

<p align="center">
 <img src = "././title.png">
</p>


This repository implements a research topic generation framework designed for YAICON, a conference organized by [YAI](<https://github.com/yonsei-YAI>) at Yonsei University.   The framework was developed to help generate high-quality research topics across various AI subfields, supporting students in ideating innovative and impactful research ideas. 

It builds upon the baseline work found in [this repository](<https://github.com/NoviScl/AI-Researcher/tree/main>), which provides the foundation for generating novel research ideas using AI.

The system accepts a research domain or theme described in natural language as input and outputs a ranked list of potential project proposals. Each proposal provides a detailed research topic, enabling students to select a direction for their project with minimal additional effort.


<p align="center">
 <img src = "./overview.png">
</p>

Our agent pipeline consists of the following modules:
(1) Related Project Retrieval;
(2) Idea Generation;
(3) Idea Deduplication;
(4) Idea Conceretization;
(5) Idea Ranking;

These modules are designed to be run sequentially as an end-to-end idea generation pipeline. 

## Overview
<p align="center">
 <img src = "./method.png">
</p>

1. [Setup](#setup)
2. [Related Project Retrieval](#related-paper-search)
3. [Idea Generation](#grounded-idea-generation)
4. [Idea Deduplication](#idea-deduplication)
5. [Idea Conceretization](#project-proposal-generation)
6. [Idea Ranking](#project-proposal-ranking)


## Setup



## Related Project Retrieval


The related project search module will propose search queries and search through the github API.   
Example usage (finding related papers for a given topic):
```

```



## Idea Generation


The idea generation module takes a topic category and a list of relevant papers as input, and returns a list of generated ideas as the output including title, problem, motivation, proposed approach, and plan.

Example usage: 
```

```


## Idea Deduplication

We do a round of deduplication to remove similar ideas generated by the grounded idea generation module.  
Compute embeddings with all-mpnet-base-v2, find idea pairs with cosine similarity > 0.75 where i ≠ j and either i ≤ N or j ≤ N,  
then remove redundant ideas: prioritize removing newly generated ones or those with higher summed similarity.

Example usage:
```

```

## Idea Conceretization


Next, we expand each seed idea into a detailed idea includng problem, motiation, proposed approach, step-by-step experiment plan, expected outcomes, and fallback plan.

Example usage:
```

```

## Idea Ranking

We rank all the generated project proposals by using an Swiss-system tournament.
The prompt includes persona assignment, YAICON guideline, evaluation criteria, instruction, and output format.

Example usage:
```

```

