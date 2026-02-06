import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline

from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from scikit_mol.safeinference import SafeInferenceWrapper
from scikit_mol.utilities import set_safe_inference_mode

from .fixtures import (
    SLC6A4_subset,
    invalid_smiles_list,
    skip_pandas_output_test,
    smiles_list,
)


def equal_val(value, expected_value):
    try:
        if np.isnan(expected_value):
            return np.isnan(value)
        else:
            return value == expected_value
    except TypeError:
        return value == expected_value


@pytest.fixture(params=[1, 2])
def transformer(request):
    return MorganFingerprintTransformer(fpSize=5, n_jobs=request.param)


@pytest.fixture(params=[np.nan, None, np.inf, 0, -100])
def smiles_pipeline(request, transformer):
    return Pipeline(
        [
            ("s2m", SmilesToMolTransformer()),
            ("FP", transformer),
            (
                "RF",
                SafeInferenceWrapper(
                    RandomForestRegressor(n_estimators=3, random_state=42),
                    replace_value=request.param,
                ),
            ),
        ]
    )


@pytest.fixture
def smiles_pipeline_trained(smiles_pipeline, SLC6A4_subset):
    X_smiles, Y = SLC6A4_subset.SMILES, SLC6A4_subset.pXC50
    X_smiles = X_smiles.to_frame()
    # Train the model
    smiles_pipeline.fit(X_smiles, Y)
    return smiles_pipeline


def test_safeinference_wrapper_basic(smiles_pipeline, SLC6A4_subset):
    X_smiles, Y = SLC6A4_subset.SMILES, SLC6A4_subset.pXC50
    X_smiles = X_smiles.to_frame()

    # Set safe inference mode
    set_safe_inference_mode(smiles_pipeline, True)

    # Train the model
    smiles_pipeline.fit(X_smiles, Y)

    # Test prediction
    predictions = smiles_pipeline.predict(X_smiles)

    assert len(predictions) == len(X_smiles)
    assert not np.any(
        equal_val(predictions, smiles_pipeline.named_steps["RF"].replace_value)
    )


def test_safeinference_wrapper_with_single_invalid_smiles(smiles_pipeline_trained):
    set_safe_inference_mode(smiles_pipeline_trained, True)
    replace_value = smiles_pipeline_trained.named_steps["RF"].replace_value
    # Test prediction
    prediction = smiles_pipeline_trained.predict(["invalid_smiles"])
    assert len(prediction) == 1
    assert equal_val(prediction[0], replace_value)


def test_safeinference_wrapper_with_invalid_smiles(
    smiles_pipeline, SLC6A4_subset, invalid_smiles_list
):
    X_smiles, Y = SLC6A4_subset.SMILES[:100], SLC6A4_subset.pXC50[:100]
    X_smiles = X_smiles.to_frame()

    # Set safe inference mode
    set_safe_inference_mode(smiles_pipeline, True)

    # Train the model
    smiles_pipeline.fit(X_smiles, Y)
    replace_value = smiles_pipeline.named_steps["RF"].replace_value
    # Create a test set with invalid SMILES
    X_test = pd.DataFrame({"SMILES": X_smiles["SMILES"].tolist() + invalid_smiles_list})
    len_invalid = len(invalid_smiles_list)
    # Test prediction with invalid SMILES
    predictions = smiles_pipeline.predict(X_test)
    invalid_predictions = predictions[-len_invalid:]
    assert len(predictions) == len(X_test)
    assert np.all(equal_val(invalid_predictions, replace_value))


def test_safeinference_wrapper_without_safe_mode(
    smiles_pipeline, SLC6A4_subset, invalid_smiles_list
):
    X_smiles, Y = SLC6A4_subset.SMILES[:100], SLC6A4_subset.pXC50[:100]
    X_smiles = X_smiles.to_frame()

    # Ensure safe inference mode is off (default behavior)
    set_safe_inference_mode(smiles_pipeline, False)

    # Train the model
    smiles_pipeline.fit(X_smiles, Y)

    # Create a test set with invalid SMILES
    X_test = pd.DataFrame({"SMILES": X_smiles["SMILES"].tolist() + invalid_smiles_list})

    # Test prediction with invalid SMILES
    with pytest.raises(Exception):
        smiles_pipeline.predict(X_test)


@skip_pandas_output_test
def test_safeinference_wrapper_pandas_output(
    smiles_pipeline, SLC6A4_subset, pandas_output
):
    X_smiles = SLC6A4_subset.SMILES[:100].to_frame()

    # Set safe inference mode
    set_safe_inference_mode(smiles_pipeline, True)

    # Fit and transform (up to the FP step)
    result = smiles_pipeline[:-1].fit_transform(X_smiles)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == len(X_smiles)
    assert result.shape[1] == smiles_pipeline.named_steps["FP"].fpSize


@skip_pandas_output_test
def test_safeinference_wrapper_get_feature_names_out(smiles_pipeline):
    # Get feature names from the FP step
    feature_names = smiles_pipeline.named_steps["FP"].get_feature_names_out()
    assert len(feature_names) == smiles_pipeline.named_steps["FP"].fpSize
    assert all(isinstance(name, str) for name in feature_names)


# Tests for return_std functionality with GaussianProcessRegressor


@pytest.fixture
def gp_pipeline():
    """Pipeline with GaussianProcessRegressor that supports return_std."""
    return Pipeline(
        [
            ("s2m", SmilesToMolTransformer()),
            ("FP", MorganFingerprintTransformer(fpSize=64)),
            (
                "GP",
                SafeInferenceWrapper(
                    GaussianProcessRegressor(kernel=RBF(), random_state=42),
                    replace_value=np.nan,
                ),
            ),
        ]
    )


@pytest.fixture
def gp_pipeline_trained(gp_pipeline, SLC6A4_subset):
    X_smiles = SLC6A4_subset.SMILES[:50].to_frame()
    Y = SLC6A4_subset.pXC50[:50]
    set_safe_inference_mode(gp_pipeline, True)
    gp_pipeline.fit(X_smiles, Y)
    return gp_pipeline


def test_safeinference_wrapper_return_std_basic(gp_pipeline_trained, SLC6A4_subset):
    """Test that return_std=True returns mean and std."""
    X_test = SLC6A4_subset.SMILES[:10].to_frame()

    # Test prediction with return_std=True
    result = gp_pipeline_trained.predict(X_test, return_std=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    y_pred, y_std = result
    assert len(y_pred) == len(X_test)
    assert len(y_std) == len(X_test)
    assert not np.any(np.isnan(y_pred))
    assert not np.any(np.isnan(y_std))
    assert np.all(y_std >= 0)  # std should be non-negative


def test_safeinference_wrapper_return_std_with_invalid(
    gp_pipeline_trained, SLC6A4_subset, invalid_smiles_list
):
    """Test return_std=True with invalid SMILES returns NaN for invalid rows."""
    valid_smiles = SLC6A4_subset.SMILES[:5].tolist()
    X_test = pd.DataFrame({"SMILES": valid_smiles + invalid_smiles_list})

    result = gp_pipeline_trained.predict(X_test, return_std=True)

    assert isinstance(result, tuple)
    y_pred, y_std = result
    assert len(y_pred) == len(X_test)
    assert len(y_std) == len(X_test)

    # Valid predictions should not be NaN
    assert not np.any(np.isnan(y_pred[: len(valid_smiles)]))
    assert not np.any(np.isnan(y_std[: len(valid_smiles)]))

    # Invalid predictions should be NaN
    assert np.all(np.isnan(y_pred[len(valid_smiles) :]))
    assert np.all(np.isnan(y_std[len(valid_smiles) :]))


def test_safeinference_wrapper_return_std_all_invalid(gp_pipeline_trained):
    """Test return_std=True when all inputs are invalid."""
    X_test = pd.DataFrame({"SMILES": ["invalid1", "invalid2"]})

    result = gp_pipeline_trained.predict(X_test, return_std=True)

    assert isinstance(result, tuple)
    y_pred, y_std = result
    assert len(y_pred) == 2
    assert len(y_std) == 2
    assert np.all(np.isnan(y_pred))
    assert np.all(np.isnan(y_std))


def test_safeinference_wrapper_return_cov_basic(gp_pipeline_trained, SLC6A4_subset):
    """Test that return_cov=True returns mean and covariance."""
    X_test = SLC6A4_subset.SMILES[:5].to_frame()

    result = gp_pipeline_trained.predict(X_test, return_cov=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    y_pred, y_cov = result
    assert len(y_pred) == len(X_test)
    assert y_cov.shape == (len(X_test), len(X_test))


def test_safeinference_wrapper_return_std_without_safe_mode(
    gp_pipeline, SLC6A4_subset, invalid_smiles_list
):
    """Test return_std=True without safe mode raises error on invalid input."""
    X_smiles = SLC6A4_subset.SMILES[:50].to_frame()
    Y = SLC6A4_subset.pXC50[:50]

    # Train without safe mode
    set_safe_inference_mode(gp_pipeline, False)
    gp_pipeline.fit(X_smiles, Y)

    # Valid input should work
    X_valid = SLC6A4_subset.SMILES[:5].to_frame()
    result = gp_pipeline.predict(X_valid, return_std=True)
    assert isinstance(result, tuple)

    # Invalid input should raise
    X_invalid = pd.DataFrame({"SMILES": invalid_smiles_list})
    with pytest.raises(Exception):
        gp_pipeline.predict(X_invalid, return_std=True)
