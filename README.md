# Sample_And_Interpret_ABSA

Run main.py to obtain graph representation of LCR-Rot-hop++.

To run LIME:
Run Lime.py for the output of Lime.
To switch LIME from SS to SSb, scroll down in Lime.py to "if __name__ == '__main__':" and uncomment "main_pos()" for SS or "main_pos_balanced()" for SSb.

To run LORE:
Run Counterfactuals.py for the output of LORE, including the counterfactual and decision rules.
To switch from SS to SSb, scroll down in Counterfactuls.py to "if __name__ == '__main__':" and uncomment "main_pos()" for SS or "main_pos_balanced_2()" for SSb.

Sensitiviy analysis:
in BERT_pert.py, in method get_perturbations(), change the parameter "proba_change" of function call of perturb_sentence() to values from "0.1" to "1." to perform sensitivity analysis on the probability of changing a word in x. 
work
