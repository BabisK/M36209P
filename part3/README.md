#Exercises on “Sequence Labeling”
######Ion Androutsopoulos, 2015–16

3. We wish to use a Viterbi decoder to find the most probable correct words t_1^k slide 30 (generalization for type-2 errors) of Part 1 (n-gram 
language models, spelling correction, and text normalization). Use a bigram (or optionally trigram) language model.

1. Draw the lattice of the Viterbi decoder in this case. What would the nodes of the lattice in each column stand for?
2. Write down the formulae to compute V_j(t_j) ach node of each column of the lattice. Explain how you obtained the formulae.
3. Extend the code you wrote for exercises 2 (Levenshtein distance) and 4 (language model) of Part 1 to develop a context-sensitive spelling corrector 
(for both type-1 and type-2 errors) that uses Levenshtein distance, a bigram (or optionally trigram) language model, and a Viterbi decoder. Train the 
language model of your spelling corrector as in exercise 4 of Part 1. [TEA course: You may use an existing implementation of Levenshtein distance.]
4. Introduce random spelling errors in the test dataset that you used in exercise 4 of Part 1, by randomly replacing characters and/or entire words by 
randomly selected ones. Report in detail how you constructed the new test dataset (that contains spelling errors) and provide appropriate statistics 
(e.g., number of characters and tokens in the new test dataset, percentage of wrong characters and tokens). Devise appropriate evaluation measures and 
baselines. Use them to evaluate your context-sensitive spelling corrector on the new test dataset. Report your evaluation measures, baselines, and 
evaluation results.
