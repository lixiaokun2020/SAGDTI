# SAGDTI: self-attention and graph neural network with multiple information representations for the prediction of drug-target interactions
Accurate identification of target proteins that interact with drugs is a vital step in silico, which can significantly foster the development of drug repurposing and drug discovery. In recent years, numerous deep learning-based methods have been introduced to treat drug-target interaction (DTI) prediction as a classification task. The output of this task is binary identification suggesting the absence or occurrence of interactions. However, existing studies often (i) neglect the unique molecular attributes when embedding drugs and proteins, and (ii) determine the interaction of drug-target pairs without considering biological interactive information. In this study, we propose an end-to-end attention-derived method based on the self-attention mechanism and graph neural network, termed SAGDTI. The aim of this method is to overcome the aforementioned drawbacks in the identification of DTI interaction. SAGDTI is the first method to sufficiently consider the unique molecular attribute representations for both drugs and targets in the input form of the Simplified Molecular Input Line Entry System (SMILES) sequences and three-dimensional structure graphs. In addition, our method innovatively aggregates the feature attributes of interactive information between drugs and targets through multi-scale topologies and diverse connections among biological entities. Experimental results on three benchmark datasets illustrate that SAGDTI outperforms existing prediction models, which benefit from the unique molecular attributes embedded by atom-level attention and biological interactive information representation aggregated by node-level attention. Moreover, a case study on severe acute respiratory syndrome coronavirus (SARS-CoV-2) shows that our model is a powerful tool for identifying DTI interactions in real life.

## Architecture
<p align="center">
<img src="https://github.com/lixiaokun2020/SAGDTI/blob/main/FlowChart.jpg" align="middle" height="80%" width="80%" />
</p>



# Acknowledgments
The authors sincerely hope to recieve any suggestions from you!

