from pathlib import Path


class Files:
    def __init__(self, running_file: str, filename: str) -> None:
        self.running_file: Path = Path(running_file)
        self.filename: str = filename + "_rerun"

        self.base = self.running_file.parent

        self.output_dir = Path("/data/gent/489/vsc48953/ML_enhance") / (self.running_file.stem + "_rerun") / "results"
        self.log_dir = self.base / "logs"

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_df_file(self, df_file_name: str) -> Path:
        return self.base.parent / df_file_name

    @property
    def SPLITS_FILE(self) -> Path:
        return Path("../splits.pkl")

    @property
    def RDKIT_FILE(self) -> Path:
        return self.base.parent / "rdkit_feature_names.json"

    @property
    def LOG_FILE(self) -> Path:
        return self.log_dir / f"{self.filename}.log"

    @property
    def LIGHTNING_LOG_DIR(self) -> Path:
        return self.log_dir / f"{self.filename}_log"

    @property
    def RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_results.pkl"

    @property
    def RESULTS_FILE_JSON(self) -> Path:
        return self.output_dir / f"{self.filename}_results.json"

    @property
    def RESULTS_FILE_MODEL(self) -> Path:
        return self.output_dir / f"{self.filename}_model.pt"

    @property
    def PFI_RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_PFI_results.csv"

    @property
    def SHAP_RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_SHAP_results.csv"
