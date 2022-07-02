# Sample_And_Interpret_ABSA

The steps mentioned to run the HAABSA++ code https://github.com/mtrusca/HAABSA_PLUS_PLUS need to be followed. Some files that are downloaded to be able to run HAABSA++ and the virtual environment of HAABSA++ are needed to run this project.

Next, download this GitHub project. Move all the contents from this project's programGeneratedData folder into the HAABSA++ programGeneratedData folder, replacing all the duplicate files from HAABSA++ with this project's files. You can now delete this project's data folder and move the entire data folder from HAABSA++ into this project.

The virtual environment created for HAABSA++ can be used to run this project.

Now that the file set up is finished and the virtual environment is ready, run main.py. This function will create the folder "trainedModelMaria" where the graph representation of LCR-Rot-hop++ ran for our data is saved. This graph is needed to run LIME and LORE.

To run LIME:

Open the data folder and create an empty folder named Lime. Lime will save its output in a few .txt files here.
Run Lime.py for the output of Lime.
To switch LIME from SS to SSb, scroll down in Lime.py to "if __name__ == '__main__':" and uncomment "main_pos()" for SS or "main_pos_balanced()" for SSb.

To run LORE:

Open the data folder and create an empty folder named Counterfactuals. LORE will save its output in a few .txt files here.
Run Counterfactuals.py for the output of LORE, including the counterfactual and decision rules.
To switch from SS to SSb, scroll down in Counterfactuls.py to "if __name__ == '__main__':" and uncomment "main_pos()" for SS or "main_pos_balanced_2()" for SSb.

To run Sensitiviy Analysis:

In BERT_pert.py, in method get_perturbations(), change the parameter "proba_change" of function perturb_sentence() to values from "0.1" to "1." to perform sensitivity analysis on the probability of changing a word in x. 
