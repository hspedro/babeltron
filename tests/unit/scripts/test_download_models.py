import pytest
from unittest.mock import patch, MagicMock
from babeltron.scripts.download_models import (
    download_model,
    parse_args,
    main,
)


class TestDownloadModels:
    @patch("babeltron.scripts.download_models.argparse.ArgumentParser.parse_args")
    def test_parse_args_default(self, mock_parse_args):
        mock_args = MagicMock()
        mock_args.size = "418M"
        mock_args.output_dir = "babeltron/model"
        mock_parse_args.return_value = mock_args

        args = parse_args()

        assert args.size == "418M"
        assert args.output_dir == "babeltron/model"

    @patch("babeltron.scripts.download_models.M2M100ForConditionalGeneration.from_pretrained")
    @patch("babeltron.scripts.download_models.M2M100Tokenizer.from_pretrained")
    @patch("babeltron.scripts.download_models.Path.mkdir")
    def test_download_model_success(self, mock_mkdir, mock_tokenizer, mock_model):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        output_dir = "test_output_dir"
        result = download_model(model_size="418M", output_dir=output_dir)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_model.assert_called_once_with("facebook/m2m100_418M")
        mock_tokenizer.assert_called_once_with("facebook/m2m100_418M")
        mock_model_instance.save_pretrained.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once()
        assert result == output_dir

    @patch("babeltron.scripts.download_models.M2M100ForConditionalGeneration.from_pretrained")
    @patch("babeltron.scripts.download_models.M2M100Tokenizer.from_pretrained")
    @patch("babeltron.scripts.download_models.Path.mkdir")
    @patch("babeltron.scripts.download_models.Path.exists")
    def test_download_model_no_symlink_needed(
        self, mock_exists, mock_mkdir, mock_tokenizer, mock_model
    ):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Simulate pytorch_model.bin already exists
        mock_exists.return_value = True

        output_dir = "test_output_dir"
        result = download_model(model_size="418M", output_dir=output_dir)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_model.assert_called_once_with("facebook/m2m100_418M")
        mock_tokenizer.assert_called_once_with("facebook/m2m100_418M")
        mock_model_instance.save_pretrained.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once()
        assert result == output_dir

    @patch("babeltron.scripts.download_models.M2M100ForConditionalGeneration.from_pretrained")
    @patch("babeltron.scripts.download_models.M2M100Tokenizer.from_pretrained")
    @patch("babeltron.scripts.download_models.Path.mkdir")
    def test_download_model_no_bin_files(self, mock_mkdir, mock_tokenizer, mock_model):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        output_dir = "test_output_dir"
        result = download_model(model_size="418M", output_dir=output_dir)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_model.assert_called_once_with("facebook/m2m100_418M")
        mock_tokenizer.assert_called_once_with("facebook/m2m100_418M")
        mock_model_instance.save_pretrained.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once()
        assert result == output_dir

    def test_download_model_invalid_size(self):
        with pytest.raises(ValueError) as excinfo:
            download_model(model_size="invalid_size")
        assert "Model size must be one of" in str(excinfo.value)

    @patch("babeltron.scripts.download_models.download_model")
    @patch("babeltron.scripts.download_models.parse_args")
    def test_main_success(self, mock_parse_args, mock_download_model):
        mock_args = MagicMock()
        mock_args.size = "418M"
        mock_args.output_dir = "test_output_dir"
        mock_parse_args.return_value = mock_args

        mock_download_model.return_value = "test_output_dir"

        result = main()

        mock_parse_args.assert_called_once()
        mock_download_model.assert_called_once_with(
            model_size="418M", output_dir="test_output_dir"
        )
        assert result == 0

    @patch("babeltron.scripts.download_models.download_model")
    @patch("babeltron.scripts.download_models.parse_args")
    def test_main_exception(self, mock_parse_args, mock_download_model):
        mock_args = MagicMock()
        mock_args.size = "418M"
        mock_args.output_dir = "test_output_dir"
        mock_parse_args.return_value = mock_args

        mock_download_model.side_effect = Exception("Test error")

        result = main()

        mock_parse_args.assert_called_once()
        mock_download_model.assert_called_once()
        assert result == 1
