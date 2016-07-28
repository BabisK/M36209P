from nltk import CFG, PCFG
from nltk.grammar import is_nonterminal
from nltk.tokenize import word_tokenize
import argparse

def cky(sentence, grammar):
    tokens = word_tokenize(sentence)
    ts = '[0]'
    for i, token in enumerate(tokens):
        ts += ' ' + token + ' [{}]'.format(i +1)
    print(ts)
    table = [[[] for i in range(len(tokens) + 1)] for j in range(len(tokens) + 1)]
    for i, token in enumerate(tokens):
        productions = grammar.productions(rhs=token)
        for prod in productions:
            table[i][i + 1].append(prod.lhs())

    for span in range(2, len(tokens) + 1):
        for start in range(len(tokens) - span + 1):
            end = start + span
            for split in range(start + 1, end):
                non_term1 = table[start][split]
                non_term2 = table[split][end]
                for nt1 in non_term1:
                    for nt2 in non_term2:
                        prodlist = grammar.productions(rhs=nt1)
                        for prod in prodlist:
                            if prod.rhs() == (nt1, nt2):
                                table[start][end].append(prod.lhs())
                                print('[{}] {} [{}] {} [{}] -> [{}] {} [{}]'.format(start, nt1, split, nt2, end, start, prod.lhs(), end))

    if grammar.start() in table[0][len(tokens)]:
        print('The sentence is derived from the grammar')
        return True
    else:
        print('The sentence is not derived from the grammar')
        return False


def pcky(sentence, grammar):
    tokens = word_tokenize(sentence)
    ts = '[0]'
    for i, token in enumerate(tokens):
        ts += ' ' + token + ' [{}]'.format(i + 1)
    print(ts)
    non_terminal = set([prod.lhs() for prod in grammar.productions() if is_nonterminal(prod.lhs())])
    table = [[{nt: 0 for nt in non_terminal} for i in range(len(tokens) + 1)] for j in range(len(tokens) + 1)]
    for i, token in enumerate(tokens):
        productions = grammar.productions(rhs=token)
        for prod in productions:
            table[i][i + 1][prod.lhs()] = prod.prob()

    for span in range(2, len(tokens) + 1):
        for start in range(len(tokens) - span + 1):
            end = start + span
            for split in range(start + 1, end):
                non_term1 = table[start][split]
                non_term2 = table[split][end]
                for nt1 in non_term1:
                    for nt2 in non_term2:
                        if non_term1[nt1] > 0 and non_term2[nt2] > 0:
                            prodlist = grammar.productions(rhs=nt1)
                            for prod in prodlist:
                                if prod.rhs() == (nt1, nt2):
                                    table[start][end][prod.lhs()] = prod.prob() * non_term1[nt1] * non_term2[nt2]
                                    print('[{}] {}:({:.2f}) [{}] {}:({:.2f}) [{}] -> [{}] {}:({:.5f}) [{}]'.format(start, nt1,
                                                                                                       non_term1[nt1],
                                                                                                       split, nt2,
                                                                                                       non_term2[nt2],
                                                                                                       end,
                                                                                                       start,
                                                                                                       prod.lhs(),
                                                                                                       table[start][end][prod.lhs()],
                                                                                                       end))

    if table[0][len(tokens)][grammar.start()] > 0:
        print('The sentence is derived from the grammar')
        return True
    else:
        print('The sentence is not derived from the grammar')
        return False


def main():
    parser = argparse.ArgumentParser(description='CKY and PCKY')
    parser.add_argument('-g', '--grammar', help='Input file name', required=True)
    parser.add_argument('-s', '--sentence', help='Input sentence', required=True)
    args = parser.parse_args()

    grammar_text = None
    with open(args.grammar, 'r') as f:
        grammar_text = f.read()

    grammar = None
    result = None
    try:
        grammar = CFG.fromstring(grammar_text)
    except ValueError:
        grammar = PCFG.fromstring(grammar_text)

    if type(grammar) is CFG:
        result = cky(args.sentence, grammar)
    elif type(grammar) is PCFG:
        result = pcky(args.sentence, grammar)

if __name__ == "__main__":
    main()
