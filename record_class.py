
import os
import re

# set this to your project root - everything else is derived from here
PROJECT_DIR = '/path/to/your/project'
WEIGHT_FOLDER = os.path.join(PROJECT_DIR, 'model_log_dir')
os.makedirs(WEIGHT_FOLDER, exist_ok=True)
RESULT_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


class Record:
    def __init__(self, experiment_identifier='WB',
                 experiment_identifier_long='basemodel-CSF_wd_1e-5_norma_z_norm_lr_0.0001_10_GB_scan',
                 weight_dir=None,
                 ):
        self.experiment_identifier = experiment_identifier
        self.experiment_identifier_long = experiment_identifier_long
        self.result_dir = os.path.join(
            RESULT_DIR, experiment_identifier, experiment_identifier_long)
        os.makedirs(self.result_dir, exist_ok=True)

        self.fuma_dir = os.path.join(
            self.result_dir, 'fuma')
        os.makedirs(self.fuma_dir, exist_ok=True)

        self.weight_dir = os.path.join(WEIGHT_FOLDER, weight_dir)

        self.PROJECT_DIR = PROJECT_DIR
        self.RESULT_DIR = os.path.join(self.PROJECT_DIR, 'GWAS_DIR')
        self.TORCH_GWAS_DIR = os.path.join(
            self.PROJECT_DIR, 'GWAS_DIR', 'torchgwas')
        # torchgwas files
        self.fastgwa_sum_stat_template = os.path.join(
            self.TORCH_GWAS_DIR, 'T1_model128_QT0_discovery.fastGWA')
        self.covarQ_related_removed = os.path.join(
            self.TORCH_GWAS_DIR, 'covarQ_related_removed.npy')
        self.sample_mask = os.path.join(
            self.TORCH_GWAS_DIR, 'sample_mask.npy')

        # index of samples used during embedding extraction
        self.sample_index_path = os.path.join(
            self.RESULT_DIR, 'Embeddings', 'sample_index.pkl')

        self.get_embedding_dir()
        self.get_torchgwas_dir()

    def get_embedding_dir(self):
        self.embedding_dir = os.path.join(self.RESULT_DIR, 'Embeddings')

    def get_torchgwas_dir(self):
        self.torchgwas_dir = os.path.join(
            self.TORCH_GWAS_DIR, self.experiment_identifier, self.experiment_identifier_long)

    def get_torchgwas_sumstat_path_dict(self):
        if not hasattr(self, 'epoch_paths_dict'):
            pattern = fr"{self.experiment_identifier_long}_epoch_(\d+)$"
            self.epoch_paths_dict = {int(re.search(pattern, f).group(1)): os.path.join(self.torchgwas_dir, f)
                                     for f in os.listdir(self.torchgwas_dir)
                                     if re.search(pattern, f) is not None}
            self.epoch_paths_dict = dict(
                sorted(self.epoch_paths_dict.items(), key=lambda x: x[0]))

        return self.epoch_paths_dict

    def get_weight_dict(self):
        if self.weight_dir is None:
            return {}

        else:
            if not hasattr(self, 'weight_path_dict'):
                pattern = fr"{self.experiment_identifier_long}epoch=(\d+)"
                self.weight_path_dict = {int(re.search(pattern, f).group(1)): os.path.join(self.weight_dir, f)
                                         for f in os.listdir(self.weight_dir)
                                         if re.search(pattern, f) is not None}

                self.weight_path_dict = dict(
                    sorted(self.weight_path_dict.items(), key=lambda x: x[0]))

            return self.weight_path_dict


class GB(Record):
    def __init__(self, experiment_identifier='WB',
                 experiment_identifier_long='basemodel-CSF_wd_1e-5_norma_z_norm_lr_0.0001_10_GB_scan',
                 weight_dir='T1_128_CSF'
                 ):
        super().__init__(experiment_identifier, experiment_identifier_long, weight_dir)
        self.combined_minp_df_path = os.path.join(
            self.result_dir, f'scan_{self.experiment_identifier.lower()}_last.pkl')
