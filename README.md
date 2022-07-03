# SI-ABSA

SI-ABSA stands for Sample and Interpret Aspect-Based Sentiment-Analysis, as the sampling
methods and the interpretability algorithms made for ABSA are the two components we focus
on in this paper.

The authors are partially suported by the work of Stefan Lam, Yin Liu, Jasper van der Vos,
and Flavius Frasincar. Their work was used to run LIME with SS and LORE with SS. You can
find their source code at https://github.com/StefanLam99/Explaining_ABSA.

The steps mentioned to run the HAABSA++ code, https://github.com/mtrusca/HAABSA_
PLUS_PLUS, need to be followed. Files which are downloaded and generated when running
HAABSA++ and the virtual environment of HAABSA++ are needed to run SI-ABSA.

After downloading SI-ABSA, https://github.com/VladMiron00/SI-ABSA, move all the
contents from this project’s “programGeneratedData” folder into the HAABSA++ “program-
GeneratedData” folder, replacing all the duplicate files from the HAABSA++ folder with the
files from SI-ABSA. You can now delete SI-ABSA’s “data” folder and move the entire “data”
folder from HAABSA++ into SI-ABSA. Note that the file 768remainingtestdata2016 original.txt
corresponds to “remaining test data”, the complete dataset containing 248 sentences. The
dataset we used in this paper is the reduced yet similarly balanced “used test data” of 25 in-
stances, found as 768remainingtestdata2016.txt. Inside SI-ABSA’s main folder, create a folder
named “trainedModelMaria”. Inside “trainedModelMaria” create another folder named “2016”.

The virtual environment created to run HAABSA++ can be used to run SI-ABSA.

Now that the file set up is finished and the virtual environment is ready, run main.py.
This class will generate inside “trainedModelMaria/2016” the graph representation of LCR-
Rot-hop++ ran for our data. This graph is needed to run LIME and LORE.

To run LIME:

Open the “data” folder and create an empty folder named “Lime”. LIME will save its output
in a few “.txt” files in this folder. Run Lime.py for the output of LIME. To switch LIME from
SS to SSb, scroll down in Lime.py to “if name == ’main’:” and uncomment “main pos()” to
run SS for LIME or “main pos balanced()” to run SSb for LIME.

To run LORE:

Open the “data” folder and create an empty folder named “Counterfactuals”. LORE will
save its output in a few “.txt” files in this folder. Run Counterfactuals.py for the output of
LORE (the counterfactual and the decision rules). To switch from SS to SSb, scroll down in
Counterfactuls.py to “if name == ’main’:” and uncomment “main pos()” to run SS for LORE
or “main pos balanced 2()” to run SSb for LORE.

To run Sensitiviy Analysis:

In the class BERT pert.py, in the method get perturbations(), modify the hyperparameter
“proba change” of function perturb sentence() to values from “0.1” to “1.” to perform sensitivity
analysis on the probability of changing a word in x. After altering this hyperparameter, just
run Lime.py or Counterfactuals.py with the desired sampling method (SS or SSb) to see the
impact of the hyperparameter configuration on the sampling method’s sentiment balance and
the interpretability algorithm’s performance.
