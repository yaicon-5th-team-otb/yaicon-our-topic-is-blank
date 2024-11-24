import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from sentence_transformers import SentenceTransformer
import json
import numpy as np

def idea_to_string(idea):
    # idea: an object has key 'title', 'problem', 'motivation', 'method', 'plan'
    return f'''Title: {idea['title']}
Problem Statement: {idea['problem']}
Motivation: {idea['motivation']}
Method: {idea['method']}
Step-by-step plan: {idea['plan']}'''

def main(args):
    with open(args.input_idea) as f:
        input_idea = json.load(f)
        input_idea_str = [idea_to_string(idea) for idea in input_idea]
    if args.prev_idea is not None:
        with open(args.prev_idea) as f:
            prev_idea = json.load(f)
            prev_idea_str = [idea_to_string(idea) for idea in prev_idea]
    else:
        prev_idea = []
        prev_idea_str = []

    model = SentenceTransformer('all-mpnet-base-v2')
    input_idea_emb = model.encode(input_idea_str) # (# ideas, 768)
    prev_idea_emb = model.encode(prev_idea_str) # (# ideas, 768), (0,) if there is no idea
    if prev_idea_emb.shape[0] > 0:
        input_idea_emb = np.concatenate([input_idea_emb, prev_idea_emb], axis=0)

    sim_mat = model.similarity(input_idea_emb, input_idea_emb).numpy() # 0 (different) <-> 1 (same)
    similar_pairs = np.nonzero(sim_mat > args.threshold)
    idea_to_delete = [False] * len(input_idea)
    for a, b in zip(similar_pairs[0], similar_pairs[1]):
        if a >= len(input_idea): break # no need to think about whether previous idea is novel
        if a >= b: continue # only see upper diagonal part
        else:
            # a is always smaller than b
            # a is always one of the indices of input idea
            if b >= len(input_idea): # if b corresponds to one of previous idea
                idea_to_delete[a] = True
            else: # both a and b correspond to input idea
                novelty_a = np.sum(sim_mat[a]) # smaller the better
                novelty_b = np.sum(sim_mat[b])
                if novelty_a < novelty_b:
                    idea_to_delete[b] = True
                else:
                    idea_to_delete[a] = True
    
    unique_idea = []
    for idea, delete in zip(input_idea, idea_to_delete):
        if delete:
            continue
        else:
            unique_idea.append(idea)

    with open(args.output_idea, 'w') as f:
        json.dump(unique_idea, f, indent=4)
    
    print(f'From {len(input_idea)} input ideas, {sum(idea_to_delete)} idea deleted and {len(unique_idea)} left.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_idea', type=str, required=True, help='Path to file contains list of generated ideas in json format')
    parser.add_argument('--prev_idea', type=str, default=None, help='Path to file contains list of previous ideas in json format')
    parser.add_argument('--output_idea', type=str, required=True, help='Path to save list of deduplicated ideas')
    parser.add_argument('--threshold', type=float, default=0.65, help='Threshold of sentence bert similarity score to identify similar idea')
    args = parser.parse_args()

    assert args.input_idea.endswith('.json'), 'Path to input idea file should be .json file'
    if args.prev_idea is not None:
        assert args.prev_idea.endswith('.json'), 'Path to previous idea file should be .json file'
    assert args.output_idea.endswith('.json'), 'Path to output idea file should be .json file'

    main(args)
