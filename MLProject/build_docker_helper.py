import os

from mlflow.pyfunc.backend import PyFuncBackend


def _patched_model_installation_steps(self, copy_src, model_path, env_manager, install_mlflow):
    return _original_model_installation_steps(
        self,
        copy_src.replace("\\", "/"),
        model_path,
        env_manager,
        install_mlflow,
    )


_original_model_installation_steps = PyFuncBackend._model_installation_steps
PyFuncBackend._model_installation_steps = _patched_model_installation_steps


from mlflow.models import build_docker


build_docker(model_uri=f"runs:/{os.environ['RUN_ID']}/model", name="churn-model")