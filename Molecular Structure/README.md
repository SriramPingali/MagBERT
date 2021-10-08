# Notebook Descriptions
Graph Formatting_MS.ipynb: Extracts coordinates of all atoms from pdb files of each proteins. This data is further used to get a Molecular structure representation of a protein. And individual representations of two protein instances in a sentence are concatenated for a Molecular Structure representation of one sample.<br><br>
Graph Formatting_Text.ipynb: Parses through the text samples and extracts protein pairs in each sentence. And each sample text is converted into vector which will  be the text representation of each sample<br><br>
And thus for each sentence, we have two types of representations in two different modalities (Molecular Structure, Text). These different modality samples are shuffled and fed into the Graph-BERT training process, making it a model agnostic learning.

# Setup
Before running the python notebooks, kindly do the following steps.
1.  Download this [zip](https://drive.google.com/file/d/1YOG6CTirzwjC-S8YLXW05hG2qvYLARlx/view?usp=sharing) folder containing pdb files in Bioinfer dataset. Extract these files into "./Molecular Structure/Bio-Infer/PDB"
2.  Download this [zip](https://drive.google.com/file/d/115G1vlkL2TOncU2XdhlcVoYGRjHRMeVT/view?usp=sharing) folder containing pdb files in HPRD dataset. Extract these files into "./Molecular Structure/HPRD/PDB"
