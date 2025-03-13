import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from babeltron.scripts.download_models import (
    download_model,
    parse_args,
    main,
    MODEL_CONFIGS,
    DEFAULT_MODEL_TYPE,
)


class TestDownloadModels:
    @patch("babeltron.scripts.download_models.argparse.ArgumentParser.parse_args")
    def test_parse_args_default(self, mock_parse_args):
        mock_args = MagicMock()
        mock_args.model_type = DEFAULT_MODEL_TYPE
        mock_args.size = None
        mock_args.output_dir = "babeltron/model"
        mock_parse_args.return_value = mock_args

        args = parse_args()

        # Should use the default size for the model type
        assert args.size == MODEL_CONFIGS[DEFAULT_MODEL_TYPE]["default_size"]
        assert args.output_dir == "babeltron/model"
        assert args.model_type == DEFAULT_MODEL_TYPE

    @patch("babeltron.scripts.download_models.M2M100ForConditionalGeneration.from_pretrained")
    @patch("babeltron.scripts.download_models.M2M100Tokenizer.from_pretrained")
    @patch("babeltron.scripts.download_models.Path.mkdir")
    @patch("babeltron.scripts.download_models.Path.symlink_to")
    @patch("babeltron.scripts.download_models.Path.exists")
    def test_download_m2m100_model_success(self, mock_exists, mock_symlink, mock_mkdir, mock_tokenizer, mock_model):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Simulate symlink doesn't exist
        mock_exists.return_value = False

        output_dir = "test_output_dir"
        result = download_model(model_type="m2m100", model_size="418M", output_dir=output_dir)

        # Check that mkdir was called twice (once for output_dir, once for model_dir)
        assert mock_mkdir.call_count == 2
        # Check that the model and tokenizer were downloaded with the correct name
        mock_model.assert_called_once_with("facebook/m2m100_418M")
        mock_tokenizer.assert_called_once_with("facebook/m2m100_418M")
        # Check that the model and tokenizer were saved
        mock_model_instance.save_pretrained.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once()
        # Check that the result is the model directory path
        expected_path = str(Path(output_dir) / "m2m100-418M")
        assert result == expected_path

    @patch("babeltron.scripts.download_models.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("babeltron.scripts.download_models.AutoTokenizer.from_pretrained")
    @patch("babeltron.scripts.download_models.Path.mkdir")
    @patch("babeltron.scripts.download_models.Path.symlink_to")
    @patch("babeltron.scripts.download_models.Path.exists")
    def test_download_nllb_model_success(self, mock_exists, mock_symlink, mock_mkdir, mock_tokenizer, mock_model):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Simulate symlink doesn't exist
        mock_exists.return_value = False

        output_dir = "test_output_dir"
        result = download_model(model_type="nllb", model_size="600M", output_dir=output_dir)

        # Check that mkdir was called twice (once for output_dir, once for model_dir)
        assert mock_mkdir.call_count == 2
        # Check that the model and tokenizer were downloaded with the correct name
        mock_model.assert_called_once_with("facebook/nllb-200-distilled-600M")
        mock_tokenizer.assert_called_once_with("facebook/nllb-200-distilled-600M")
        # Check that the model and tokenizer were saved
        mock_model_instance.save_pretrained.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once()
        # Check that the result is the model directory path
        expected_path = str(Path(output_dir) / "nllb-600M")
        assert result == expected_path

    @patch("babeltron.scripts.download_models.M2M100ForConditionalGeneration.from_pretrained")
    @patch("babeltron.scripts.download_models.M2M100Tokenizer.from_pretrained")
    @patch("babeltron.scripts.download_models.Path.mkdir")
    @patch("babeltron.scripts.download_models.Path.symlink_to")
    @patch("babeltron.scripts.download_models.Path.exists")
    def test_download_model_symlink_exists(self, mock_exists, mock_symlink, mock_mkdir, mock_tokenizer, mock_model):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Simulate symlink already exists
        mock_exists.return_value = True

        output_dir = "test_output_dir"
        result = download_model(model_type="m2m100", model_size="418M", output_dir=output_dir)

        # Check that mkdir was called twice (once for output_dir, once for model_dir)
        assert mock_mkdir.call_count == 2
        # Check that symlink_to was not called since the symlink already exists
        mock_symlink.assert_not_called()
        # Check that the model and tokenizer were downloaded with the correct name
        mock_model.assert_called_once_with("facebook/m2m100_418M")
        mock_tokenizer.assert_called_once_with("facebook/m2m100_418M")
        # Check that the result is the model directory path
        expected_path = str(Path(output_dir) / "m2m100-418M")
        assert result == expected_path

    def test_download_model_invalid_model_type(self):
        with pytest.raises(ValueError) as excinfo:
            download_model(model_type="invalid_model_type")
        assert "Model type must be one of" in str(excinfo.value)

    def test_download_model_invalid_size(self):
        with pytest.raises(ValueError) as excinfo:
            download_model(model_type="m2m100", model_size="invalid_size")
        assert "size must be one of" in str(excinfo.value)

    @patch("babeltron.scripts.download_models.download_model")
    @patch("babeltron.scripts.download_models.parse_args")
    def test_main_success(self, mock_parse_args, mock_download_model):
        mock_args = MagicMock()
        mock_args.model_type = "m2m100"
        mock_args.size = "418M"
        mock_args.output_dir = "test_output_dir"
        mock_parse_args.return_value = mock_args

        mock_download_model.return_value = "test_output_dir/m2m100-418M"

        result = main()

        mock_parse_args.assert_called_once()
        mock_download_model.assert_called_once_with(
            model_type="m2m100", model_size="418M", output_dir="test_output_dir"
        )
        assert result == 0

    @patch("babeltron.scripts.download_models.download_model")
    @patch("babeltron.scripts.download_models.parse_args")
    def test_main_exception(self, mock_parse_args, mock_download_model):
        mock_args = MagicMock()
        mock_args.model_type = "m2m100"
        mock_args.size = "418M"
        mock_args.output_dir = "test_output_dir"
        mock_parse_args.return_value = mock_args

        mock_download_model.side_effect = Exception("Test error")

        result = main()

        mock_parse_args.assert_called_once()
        mock_download_model.assert_called_once_with(
            model_type="m2m100", model_size="418M", output_dir="test_output_dir"
        )
        assert result == 1
