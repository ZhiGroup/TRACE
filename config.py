import os

# set PROJECT_DIR in record_class.py - paths below should match

# path to your MRI data root
data_path = '/path/to/your/data'

# cohort CSVs - columns: participant_id, t1_mri_path, t1_brain_path, t1_seg_path
disc_path  = os.path.join(data_path, 'discovery_cohort.csv')
rep_path   = os.path.join(data_path, 'replication_cohort.csv')

train_datafile = os.path.join(data_path, 'train_split.csv')
val_datafile   = os.path.join(data_path, 'val_split.csv')
test_datafile  = os.path.join(data_path, 'test_split.csv')

# embeddings go here - should match record_class PROJECT_DIR structure
embedding_dir = '/path/to/your/project/GWAS_DIR/Embeddings'
