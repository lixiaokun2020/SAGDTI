def SAGDTI_config():
    config = {}
    config['batch_size'] = 256
    config['input_dim_drug'] = 23532
    config['input_dim_target'] = 16693
    config['max_drug_seq'] = 150
    config['max_protein_seq'] = 1200
    config['setting_no'] = 2
    config['emb_size'] = 500
    config['dropout_rate'] = 0.1
    
    #DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3
    
    # molecular transformer encoder
    config['intermediate_size'] = 500
    config['num_attention_heads'] = 8
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    config['flat_dim'] = 78192

    #biological interactive information with GAT
    config['graphdim'] = 500
    config['GAThid1'] = 256
    config['GAThid2'] = 64
    config['weight_decay'] = 1e-3
    config['GATlayers'] = 3

    config['feature_path'] = '/biological interactive information/graph/dti/dti.feature'
    config['label_path'] = '/biological interactive information/graph/dti/dti.label'
    config['graph_path'] = '/biological interactive information/graph/dti/knn/c2.txt'


    return config